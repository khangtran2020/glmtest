import os
import re
import ast
import json
import torch
import openai
import argparse
import numpy as np
from tqdm import tqdm
from rich.console import Console
from typing import List, Dict, Any, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity


# RAG Prompts with retrieved context
PROMPT_RAG = """Generate a test case for the following module such that:
- The test case uses the pytest framework and is executable.
- The test case will be put in the `tests/` directory which is placed in the root of the project.
- The test case will need to execute the provided branch of execution in the provided module.

Here is the module:
```python
{}
```

The module is from this path:
{}

Here is the execution branch. The execution branch is a sequence of executable line numbers in the module:
{}

## RETRIEVED SIMILAR EXAMPLES:
{}

Just output your answer WITHOUT REASONING and ensure your response is in the following format:

```json
{{
  "test_case": <YOUR ANSWER FOR THE TEST CASE - JUST ONLY THE EXECUTABLE PYTHON CODE>
}}
```
"""


def extract_test_case(raw: Union[str, dict]) -> str:
    """Extract test case from LLM response."""
    if isinstance(raw, dict):
        return raw.get("test_case", "")

    if isinstance(raw, str):
        # Remove markdown fences
        fence = re.compile(r"```(?:json)?\s*([\s\S]*?\{[\s\S]*?\})\s*```", re.MULTILINE)
        m = fence.search(raw)
        payload = m.group(1) if m else raw

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(payload)
            except Exception:
                return ""

        return data.get("test_case", "")

    return ""


def get_embedding(
    text: str, client: openai.OpenAI, model: str = "text-embedding-3-small"
) -> List[float]:
    """Get embedding for text using OpenAI API."""
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []


def query_openai(
    prompt: str,
    client: openai.OpenAI,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Query OpenAI API."""
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = response.choices[0].message.content
        usage = response.usage.model_dump() if response.usage else {}

        return {
            "success": True,
            "content": content,
            "usage": usage,
            "extract_content": extract_test_case(content),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": "",
            "extract_content": "",
        }


class RAGTestGenerator:
    """RAG-based test case generator using OpenAI."""

    def __init__(self, api_key: str, console: Console, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.console = console
        self.kb_docs = []
        self.kb_embeddings = []

    def build_knowledge_base(
        self, dataset: Dict[str, Any], save_path: str, split: str = "train"
    ):
        embedding_path = os.path.join(save_path, f"kb_embeddings.pt")
        if os.path.exists(embedding_path):
            self.console.log(
                f"[yellow]Loading existing knowledge base embeddings from {embedding_path}...[/yellow]"
            )
            embs_and_docs = torch.load(embedding_path, allow_pickle=True)
            self.kb_embeddings = embs_and_docs["embeddings"]
            self.kb_docs = embs_and_docs["docs"]
            self.console.log(
                f"[green]Loaded knowledge base: {len(self.kb_docs)} examples[/green]"
            )
            return

        """Build knowledge base from dataset test cases."""
        self.console.log("[yellow]Building knowledge base...[/yellow]")

        data = dataset.processed_data[split]
        for key in tqdm(data.keys()):
            uuid = data[key]["uuid"]
            module_path = data[key]["module_path"]
            code_path = data[key]["code_path"]
            with open(code_path, "r") as f:
                module_code = f.read()

            for tc_key in data[key]["test_cases"].keys():
                branch = data[key]["test_cases"][tc_key]["branch"]
                test_case = data[key]["test_cases"][tc_key]["test_case"]
                branch_line = ""
                for i, branch_item in enumerate(branch):
                    branch_line += (
                        f"Branch #{i+1}"
                        + "->".join([str(item) for item in branch_item])
                        + "\n"
                    )

                doc_text = f"### Module: {module_path}\n### Code:\n{module_code[:300]}\n### Branch: {branch_line} -> \n### Test: {test_case}"
                embedding = get_embedding(doc_text, self.client)

                if embedding:
                    self.kb_docs.append(
                        {
                            "module_path": module_path,
                            "test_case": test_case,
                            "branch": branch_line,
                            "code": module_code,
                        }
                    )
                    self.kb_embeddings.append(embedding)

        self.kb_embeddings = np.array(self.kb_embeddings)
        embs_and_docs = {"embeddings": self.kb_embeddings, "docs": self.kb_docs}
        torch.save(embs_and_docs, embedding_path)
        self.console.log(
            f"[green]Knowledge base built: {len(self.kb_docs)} examples[/green]"
        )
        self.console.log(f"[yellow]Saved embeddings to {embedding_path}[/yellow]")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k similar examples."""
        if len(self.kb_docs) == 0:
            return []

        query_emb = get_embedding(query, self.client)
        if not query_emb:
            return []

        query_emb = np.array(query_emb).reshape(1, -1)
        similarities = cosine_similarity(query_emb, self.kb_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {"doc": self.kb_docs[idx], "score": float(similarities[idx])}
            )
        return results

    def format_context(self, retrieved: List[Dict[str, Any]]) -> str:
        """Format retrieved examples."""
        if not retrieved:
            return "No examples available."

        parts = []
        for i, res in enumerate(retrieved, 1):
            doc = res["doc"]
            parts.append(
                f"Example {i}:\n### Module: {doc['module_path']}\n### Branch: {doc['branch']}\n```python\n{doc['test_case']}\n```\n"
            )

        return "\n".join(parts)

    def generate_test_cases(
        self,
        dataset: Dict[str, Any],
        output_path: str,
        db_path: str,
        output_name: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        use_rag: bool = True,
        top_k: int = 3,
    ):
        """Generate test cases with RAG."""

        # Build KB if using RAG
        if use_rag:
            self.build_knowledge_base(dataset, save_path=db_path, split="train")

        # Setup output
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{output_name}_responses.jsonl")
        test_file = os.path.join(output_path, f"{output_name}_test_cases.jsonl")

        # Load processed
        processed = set()
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        key = f"{rec['instance_id']}_{rec['branch_key']}"
                        processed.add(key)
                    except:
                        pass

        # Generate
        generated = 0
        skipped = 0

        data = dataset.data["test_module"]

        # Add tqdm progress bar
        for key in tqdm(data.keys(), desc="Generating test cases"):
            uuid = data[key]["uuid"]
            module_path = data[key]["module_path"]
            code_path = data[key]["code_path"]
            with open(code_path, "r") as f:
                module_code = f.read()

            for tc_key in data[key]["test_cases"].keys():
                branch = data[key]["test_cases"][tc_key]["branch"]
                branch_line = ""
                for i, branch_item in enumerate(branch):
                    branch_line += (
                        f"Branch #{i+1}"
                        + "->".join([str(item) for item in branch_item])
                        + "\n"
                    )

                # Retrieve if using RAG
                context = ""
                if use_rag and self.kb_embeddings.size > 0:
                    query = f"### Module: {module_path}\n### Code:\n{module_code[:300]}\n### Branch: {branch_line}"
                    retrieved = self.retrieve(query, top_k)
                    context = self.format_context(retrieved)

                # Build prompt
                prompt = PROMPT_RAG.format(
                    module_code, module_path, branch_line, context
                )

                # Query
                response = query_openai(
                    prompt, self.client, self.model, max_tokens, temperature
                )

                # Save response
                record = {
                    "instance_id": uuid,
                    "module_path": module_path,
                    "branch_key": tc_key,
                    "response": response,
                }

                with open(output_file, "a") as f:
                    f.write(json.dumps(record) + "\n")

                # Save test case
                test_id = f"{uuid}_test_case_{tc_key.split('_')[-1]}"
                with open(test_file, "a") as f:
                    f.write(
                        json.dumps({test_id: response.get("extract_content", "")})
                        + "\n"
                    )
                generated += 1

        self.console.log(
            f"[green]Done! Generated: {generated}, Skipped: {skipped}[/green]"
        )


# def load_dataset(path: str) -> Dict[str, Any]:
#     """Load dataset from JSON."""
#     with open(path, "r") as f:
#         data_list = json.load(f)

#     dataset = {}
#     for i, item in enumerate(data_list):
#         dataset[f"instance_{i}"] = item

#     return dataset


# def main():
#     parser = argparse.ArgumentParser(
#         description="RAG-based test generation with OpenAI"
#     )

#     # Required
#     parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
#     parser.add_argument(
#         "--data_path", type=str, required=True, help="Path to processed_data.json"
#     )
#     parser.add_argument(
#         "--output_path", type=str, required=True, help="Output directory"
#     )
#     parser.add_argument("--output_name", type=str, required=True, help="Output prefix")

#     # Model
#     parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model")

#     # Generation
#     parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
#     parser.add_argument("--max_tokens", type=int, default=2000, help="Max tokens")

#     # RAG
#     parser.add_argument("--use_rag", action="store_true", default=True, help="Use RAG")
#     parser.add_argument(
#         "--no_rag", dest="use_rag", action="store_false", help="Disable RAG"
#     )
#     parser.add_argument("--top_k", type=int, default=3, help="Top-k retrieval")

#     args = parser.parse_args()

#     # Load dataset
#     console = Console()
#     console.log(f"[yellow]Loading dataset from {args.data_path}...[/yellow]")
#     dataset = load_dataset(args.data_path)
#     console.log(f"[green]Loaded {len(dataset)} instances[/green]")

#     # Initialize generator
#     generator = RAGTestGenerator(api_key=args.api_key, model=args.model)

#     # Generate
#     generator.generate_test_cases(
#         dataset=dataset,
#         output_path=args.output_path,
#         output_name=args.output_name,
#         max_tokens=args.max_tokens,
#         temperature=args.temperature,
#         use_rag=args.use_rag,
#         top_k=args.top_k,
#     )


# if __name__ == "__main__":
#     main()

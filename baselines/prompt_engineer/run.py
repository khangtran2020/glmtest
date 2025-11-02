import os
import re
import ast
import json
import openai
import anthropic
from tqdm import tqdm
from rich.console import Console
from data.core import Data
from typing import List, Dict, Any, Optional, Union

PROMPT_ZERO_SHOT = """
You are an assistant that generates a single pytest test case for a given Python module and a given execution branch.

INSTRUCTIONS (must follow exactly):
1) Think internally as needed, but do NOT output any chain-of-thought, reasoning, notes, or explanation. Only output the final JSON object described below — nothing else (no text, no code fences, no comments).
2) The output MUST be exactly one JSON object with a single key "test_case" whose value is a string containing the complete, executable Python test file contents (pytest compatible).
3) If you cannot produce a valid test case, output exactly: {{"test_case": ""}} and nothing else.
4) The JSON must parse. Do not output backticks or markdown fences. Do not prepend or append any extra characters.
5) The test_case string should contain only valid Python code (with imports as needed) that, when saved into tests/, exercises the provided branch.

MODULE:
```python
{}
```

MODULE PATH:
{}

EXECUTION BRANCH (a sequence of executable line numbers or segments):
{}

Output ONLY the final JSON object now, for example:
{{ "test_case": "import ...\n\ndef test_...():\n    ..." }}
"""

PROMPT_COT = """
You are an assistant that will produce a single pytest test case for a given Python module and a given execution branch.

IMPORTANT:
- You may think privately/internal to yourself, but you MUST NOT output any chain-of-thought or intermediate reasoning.
- Your ONLY visible output must be a single JSON object with exactly one key "test_case". The value must be a string containing the full Python test (pytest) code.
- If you cannot produce a valid test, return exactly: {{"test_case": ""}}.
- No extra text, no explanations, and no code fences — only the JSON object.

MODULE:
```python
{}
```

MODULE PATH:
{}

EXECUTION BRANCH:
{}

Now, after private consideration, output the final JSON object only:
{{ "test_case": "<PLACE YOUR PYTEST CODE HERE>" }}
"""

KEY_TEMPLATE = "{}_{}"


def extract_test_case(raw: Union[str, dict]) -> str:
    """
    Given either:
      - A dict with a 'test_case' key,
      - A JSON string (possibly wrapped in ```json …``` fences), or
      - A Python‐literal dict string,
    this will return the inner multi‐line code.
    """
    # 1) If it's already a dict, just pull it out:
    if isinstance(raw, dict):
        return raw["test_case"]

    # 2) If it's a string, strip any Markdown fences around the JSON:
    if isinstance(raw, str):
        # look for ```json … { … } … ```
        fence = re.compile(r"```(?:json)?\s*([\s\S]*?\{[\s\S]*?\})\s*```", re.MULTILINE)
        m = fence.search(raw)
        payload = m.group(1) if m else raw

        # 3) Try JSON first:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            # 4) Fallback to Python literal (single‐ or double‐quoted):
            try:
                data = ast.literal_eval(payload)
            except Exception as e:
                print(f"Failed to parse payload: {payload}")
                raise ValueError(
                    "Could not parse payload as JSON or Python literal"
                ) from e

        if "test_case" not in data:
            raise KeyError("No 'test_case' key found")
        return data["test_case"]

    raise TypeError(f"Expected str or dict, got {type(raw)}")


# --- Modified init_api function ---
def init_api(model: str, api_key: str):
    """
    Initializes and returns the appropriate API client based on the model name.

    Args:
        model (str): The name of the model (e.g., "gpt-3.5-turbo", "claude-3-opus-20240229", "gemini-pro").
        api_key (str): The API key for the chosen service.

    Returns:
        Union[openai.OpenAI, anthropic.Anthropic, GeminiClient]: An initialized API client.

    Raises:
        ValueError: If an unsupported model is provided.
    """
    if ("o3-mini" in model) or ("gpt" in model) or ("deepseek" in model):
        client = openai.OpenAI(api_key=api_key)
    elif "claude" in model:
        client = anthropic.Anthropic(api_key=api_key)
    elif "gemini" in model:
        # Gemini models are not supported in this runtime. Raise to make behavior explicit.
        raise ValueError("Gemini models are not supported by this script")
    else:
        raise ValueError(f"Unsupported model: {model}")
    return client


# --- Modified query_prompt function ---
def query_prompt(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    client: Any,
) -> Dict[str, Any]:
    """
    Queries the respective LLM API with the given prompt and parameters.

    Args:
        prompt (str): The prompt message to send to the LLM.
        model (str): The name of the model to use.
        max_tokens (int): The maximum number of tokens to generate in the response.
        temperature (float): Controls the randomness of the output.
        client (Any): An initialized API client (OpenAI, Anthropic, or GeminiClient).

    Returns:
        Dict[str, Any]: A dictionary containing the response details.
                        Includes 'success', 'content', 'model', 'usage', and 'stop_reason'.

    Raises:
        Exception: If the API call fails for any reason.
    """
    try:
        if isinstance(client, openai.OpenAI):
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "model": model,
                "max_completion_tokens": max_tokens,
                # "temperature": temperature,
                "messages": messages,
            }
            response = client.chat.completions.create(**kwargs)

            # print out response for debugging
            print(f"Response: {response}")

            # OpenAI's chat.completions.create returns a Completion object
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                },
                "stop_reason": response.choices[0].finish_reason,
                "extract_content": extract_test_case(
                    response.choices[0].message.content
                ),
            }

        elif isinstance(client, anthropic.Anthropic):
            # Anthropic's messages.create API
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,  # Anthropic uses max_tokens, not max_output_tokens
                "temperature": temperature,
                "messages": messages,
            }
            response = client.messages.create(**kwargs)

            # Anthropic's messages.create returns a Message object
            return {
                "success": True,
                "content": response.content[0].text,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "stop_reason": response.stop_reason,
                "extract_content": extract_test_case(response.content[0].text),
            }

        # Note: Gemini client support has been removed. If you need Gemini, add a custom client implementation.

        else:
            raise TypeError("Unsupported client type provided.")

    except Exception as e:
        print(f"An error occurred during query: {e}")
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "content": str(e),
            "model": model,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "stop_reason": None,
        }


class PromptEngineer:

    def __init__(self, args, model: str, api_key: str, console: Console):
        self.args = args
        self.console = console
        self.model = model
        self.client = init_api(model=model, api_key=api_key)

    def build_prompt(
        self, dataset: Data, split: str = "test_module", prompt_type: str = "zero_shot"
    ) -> List[Dict[str, str]]:

        data = dataset.data[split]
        prompt_list = []

        # Add tqdm progress bar
        for key in tqdm(data.keys()):
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
                if prompt_type == "cot":
                    prompt_text = PROMPT_COT.format(
                        module_code, module_path, branch_line
                    )
                else:
                    prompt_text = PROMPT_ZERO_SHOT.format(
                        module_code, module_path, branch_line
                    )

                prompt_list.append(
                    {
                        "instance_id": uuid,
                        "module_path": module_path,
                        "code_path": code_path,
                        "branch_key": tc_key,
                        "prompt": prompt_text,
                    }
                )
        self.console.log(
            f"[green]Built {len(prompt_list)} prompts for prompt engineering.[/green]"
        )
        return prompt_list

    def generate_responses(
        self,
        prompt_list: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        output_path: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        responses: List[Dict[str, Any]] = []

        # Prepare output files if requested and discover already-processed keys
        output_file = None
        test_case_file = None
        processed_keys = set()
        if output_path is not None and output_name is not None:
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, f"{output_name}_responses.jsonl")
            test_case_file = os.path.join(
                output_path, f"{output_name}_test_cases.jsonl"
            )

            # If responses file exists, read processed keys to skip
            if os.path.exists(output_file):
                try:
                    with open(output_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                iid = rec.get("instance_id")
                                bkey = rec.get("branch_key")
                                if iid is not None and bkey is not None:
                                    processed_keys.add(KEY_TEMPLATE.format(iid, bkey))
                            except Exception:
                                # ignore malformed lines
                                continue
                except Exception:
                    # If reading fails, proceed without skipping
                    processed_keys = set()

        skipped = 0
        generated = 0

        for prompt_item in tqdm(prompt_list):
            iid = prompt_item["instance_id"]
            bkey = prompt_item["branch_key"]
            key = KEY_TEMPLATE.format(iid, bkey)

            if key in processed_keys:
                skipped += 1
                continue

            prompt_text = prompt_item["prompt"]
            response = query_prompt(
                prompt=prompt_text,
                client=self.client,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            response_record = {
                "instance_id": iid,
                "module_path": prompt_item["module_path"],
                "branch_key": bkey,
                "response": response,
            }

            # Append to responses list (in-memory)
            responses.append(response_record)

            # Persist immediately (append mode) so we can resume later
            if output_file is not None:
                try:
                    with open(output_file, "a") as f:
                        f.write(json.dumps(response_record) + "\n")
                except Exception as e:
                    self.console.log(
                        f"[red]Failed to append to responses file: {e}[/red]"
                    )

            # Persist extracted test case as a small mapping {uuid: test_case}
            if test_case_file is not None:
                try:
                    uuid = f"{iid}_test_case_{bkey.split('_')[-1]}"
                    extract_content = ""
                    try:
                        extract_content = response.get("extract_content", "")
                    except Exception:
                        extract_content = ""
                    test_case_record = {uuid: extract_content}
                    with open(test_case_file, "a") as f:
                        f.write(json.dumps(test_case_record) + "\n")
                except Exception as e:
                    self.console.log(
                        f"[red]Failed to append to test case file: {e}[/red]"
                    )

            # Mark as processed so we don't re-run in the same session
            processed_keys.add(key)
            generated += 1

        self.console.log(
            f"[green]Generated {generated} new responses (skipped {skipped})[/green]"
        )

        return responses

    def run_prompt_engineering(
        self,
        dataset,
        prompt_type: str,
        temperature: float,
        output_path: str,
        output_name: str,
        max_tokens: int,
    ):
        prompt_list = self.build_prompt(
            dataset=dataset,
            split="test_module",
            prompt_type=prompt_type,
        )
        self.console.log(
            f"[green]Built {len(prompt_list)} prompts for prompt engineering.[/green]"
        )

        # print one prompt for debugging
        self.console.log(f"[yellow]Sample Prompt:[/yellow]\n{prompt_list[0]['prompt']}")

        self.generate_responses(
            prompt_list=prompt_list,
            max_tokens=max_tokens,
            temperature=temperature,
            output_path=output_path,
            output_name=output_name,
        )
        self.console.log(f"[green]Generated responses and saved[/green]")

import json
import ast
import openai
import anthropic
import re


# from utils.utils import console
from typing import Dict, Any, Union

PROMPT_VERIFY = """
I have a test case that may contain hallucinated/unnecessary imports and compilation issues. Please:

1. **Analyze the imports**: Check which models and test classes are actually used in the test code. Identify and remove all unused imports.

2. **Verify compilation**: Ensure the test case will compile and run without import errors or syntax issues.

3. **Refactor the imports**: Keep only the necessary imports that are actually referenced in the test code.

4. **Preserve the test logic**: Do NOT change the test assertions or logic - keep them exactly as written, even if they seem incorrect. The goal is only to make the test runnable, not to fix its correctness.

5. **Provide the cleaned test case**: Output the refactored code with minimal, correct imports.

Here's the test case to clean:

```python
{test_case}
```

Please show only the final cleaned test case between ```python and ``` tags, without any additional explanation.
"""


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
        fence = re.compile(
            r"```(?:python)?\s*([\s\S]*?\{[\s\S]*?\})\s*```", re.MULTILINE
        )
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
    if model.startswith("o3-mini") or ("gpt" in model) or ("deepseek" in model):
        client = openai.OpenAI(api_key=api_key)
    elif "claude" in model:
        client = anthropic.Anthropic(api_key=api_key)
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
                "temperature": temperature,
                "messages": messages,
            }
            response = client.chat.completions.create(**kwargs)

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
        else:
            raise TypeError("Unsupported client type provided.")

    except Exception as e:
        print(f"An error occurred during query: {e}")
        return {
            "success": False,
            "content": str(e),
            "model": model,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "stop_reason": None,
        }


def verify_test_case(
    test_case: str,
    model: str,
    max_tokens: int,
    temperature: float,
    api_key: str,
) -> Dict[str, Any]:
    client = init_api(model=model, api_key=api_key)
    prompt = PROMPT_VERIFY.format(test_case=test_case)
    varifier_output = query_prompt(
        prompt=prompt,
        client=client,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    refactored_code = varifier_output["extract_content"]
    return {
        "refactored_code": refactored_code,
        "varifier_output": varifier_output,
    }


# def run(args):
#     print(
#         {
#             "input_file": args.input_file,
#             "output_file": args.output_file,
#             "model": args.model,
#         }
#     )
#     # api_dict = None
#     with open(args.api_file) as f:
#         api_dict = json.load(f)

#     if args.model.startswith("o3-mini") or "gpt" in args.model:
#         api_key = api_dict["gpt"]
#     elif "deepseek" in args.model:
#         api_key = api_dict["deepseek"]
#     elif "gemini" in args.model:
#         api_key = api_dict["gemini"]
#     elif "claude" in args.model:
#         api_key = api_dict["claude"]

#     client = init_api(model=args.model, api_key=api_key)

#     # read data: the json file from the input_file
#     input_data = []
#     with open(args.input_file) as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             input_data.append(json.loads(line))

#     result_dict = {}

#     # process each data point to query:
#     for inputTemp in input_data:
#         uuid = inputTemp["uuid"]
#         module = inputTemp["code_src"]
#         branch = inputTemp["branches"]
#         for item in branch:
#             key = KEY_TEMPLATE.format(uuid, item)
#             print(f"Key: {key}")
#             execution_branch = ""
#             test = branch[item]
#             for t_branch in test:
#                 item_str = "->".join(str(x) for x in t_branch)
#                 execution_branch += f"{item_str}\n"

#             prompt = PROMPT_COT.format(module=module, execution_branch=execution_branch)
#             response = query_prompt(
#                 prompt=prompt,
#                 client=client,
#                 model=args.model,
#                 temperature=args.temperature,
#                 max_tokens=args.max_tokens,
#                 reasoning=args.reasoning,
#             )
#             result_dict[key] = response

#             # write the result_dict to the output file
#             with open(args.output_file, "w") as f:
#                 json.dump(result_dict, f, indent=2)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process some prompts.")
#     parser.add_argument("--input_file", type=str, help="Path to the input file")
#     parser.add_argument("--output_file", type=str, help="Path to the output file")
#     parser.add_argument("--api_file", type=str, help="API File")
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="o3-mini-2025-01-31",
#         help="Model to use for generation",
#     )
#     parser.add_argument(
#         "--temperature", type=float, default=1, help="Temperature for generation"
#     )
#     parser.add_argument(
#         "--max_tokens", type=int, default=4096, help="Maximum tokens for generation"
#     )
#     parser.add_argument(
#         "--reasoning", type=str, default="medium", help="Reasoning Effort"
#     )
#     args = parser.parse_args()
#     run(args)

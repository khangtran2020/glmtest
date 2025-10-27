import re
import openai
import anthropic

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


def extract_code_block(text: str):
    pattern = r"```python(?:\w+)?\n([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return text


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
    # try:
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
            "extract_content": extract_code_block(response.choices[0].message.content),
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
            "extract_content": extract_code_block(response.content[0].text),
        }
    else:
        raise TypeError("Unsupported client type provided.")


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

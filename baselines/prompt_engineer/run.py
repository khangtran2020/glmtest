import os
import json
import argparse
import ast
import openai
import anthropic
import google.generativeai as genai
import re

# from utils.utils import console
from typing import List, Dict, Any, Optional, Union

PROMPT_ZERO_SHOT = """Generate a test case for the following module such that:
- The test case use the pytest framework and executable.
- The test case will be put in the `tests/` directory which is place in the root of the project.
- The test case will need to execute the provided branch of execution in the provided module.

Here is the module:
```python
{}
```

Here is the execution branch. The execution branch is a sequence of executable line number in the module:
{}

Just output your answer WITHOUT REASONING and ensure your response is in the following format:

```json
{{
  "test_case": <YOUR ANSWER FOR THE TEST CASE - JUST ONLY THE EXECUTABLE PYTHON CODE>
}}
```
"""

PROMPT_COT = """Generate a test case for the following module such that:
- The test case use the pytest framework and executable.
- The test case will be put in the `tests/` directory which is place in the root of the project.
- The test case will need to execute the provided branch of execution in the provided module.

Here is the module:
```python
{module}
```

Here is the execution branch. The execution branch is a sequence of executable line number in the module:
{execution_branch}

THINK STEP-BY-STEP and provide your response in the following format:

```json
{{
  "test_case": <YOUR ANSWER FOR THE TEST CASE - JUST ONLY THE EXECUTABLE PYTHON CODE>
}}
```
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
        return raw['test_case']

    # 2) If it's a string, strip any Markdown fences around the JSON:
    if isinstance(raw, str):
        # look for ```json … { … } … ```
        fence = re.compile(r'```(?:json)?\s*([\s\S]*?\{[\s\S]*?\})\s*```', re.MULTILINE)
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
                raise ValueError("Could not parse payload as JSON or Python literal") from e

        if 'test_case' not in data:
            raise KeyError("No 'test_case' key found")
        return data['test_case']

    raise TypeError(f"Expected str or dict, got {type(raw)}")


    
# --- GeminiClient Class (re-defined for this integration) ---
class GeminiClient:
    """
    A client for interacting with the Google Gemini API.

    Encapsulates API configuration and common generative AI operations.
    """

    def __init__(
        self, api_key: Optional[str] = None, default_model: str = "gemini-pro"
    ):
        """
        Initializes the GeminiClient.

        Args:
            api_key (str, optional): Your Google Gemini API key. If not provided,
                                     it attempts to load from the GOOGLE_API_KEY
                                     environment variable.
            default_model (str): The default Gemini model to use for operations.
                                 Defaults to "gemini-pro".
        Raises:
            ValueError: If no API key is provided or found in environment variables.
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Google API Key not provided. Please provide it as an argument "
                    "or set the GOOGLE_API_KEY environment variable."
                )

        genai.configure(api_key=api_key)
        self.default_model = default_model
        self._models = {}  # Cache models to avoid re-initializing them unnecessarily

    def _get_model(self, model_name: str) -> genai.GenerativeModel:
        """Helper to get or create a GenerativeModel instance."""
        if model_name not in self._models:
            self._models[model_name] = genai.GenerativeModel(model_name)
        return self._models[model_name]

    def generate_content(
        self, prompt: str | List[Any], model_name: Optional[str] = None, **kwargs: Any
    ) -> genai.types.GenerateContentResponse:
        """
        Generates content based on a given prompt (text or multimodal).
        This method is adjusted to return the raw API response object for more details.

        Args:
            prompt (str | List[Any]): The text prompt or a list of multimodal content parts.
            model_name (str, optional): The specific model to use for this request.
                                        Defaults to the client's default_model.
            **kwargs: Additional keyword arguments for `model.generate_content()`,
                      e.g., `temperature`, `max_output_tokens`, `safety_settings`.

        Returns:
            genai.types.GenerateContentResponse: The raw response object from the Gemini API.

        Raises:
            Exception: If the API call fails.
        """
        model_to_use = model_name if model_name else self.default_model
        model = self._get_model(model_to_use)

        try:
            # For Gemini, max_tokens is `max_output_tokens` in generation_config
            generation_config = kwargs.pop("generation_config", {})
            if "max_tokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs.pop("max_tokens")

            # The temperature parameter is also part of generation_config for Gemini
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs.pop("temperature")

            response = model.generate_content(
                prompt, generation_config=generation_config, **kwargs
            )
            return response
        except Exception as e:
            print(f"Error generating content with Gemini model '{model_to_use}': {e}")
            raise


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
    elif "gemini" in model:
        # Pass the API key to our custom GeminiClient
        client = GeminiClient(api_key=api_key)
    else:
        raise ValueError(f"Unsupported model: {model}")
    return client


# --- Modified query_prompt function ---
def query_prompt(
    prompt: str, model: str, max_tokens: int, temperature: float, client: Any, reasoning: str,
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
                "reasoning_effort": reasoning,
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
                "extract_content": extract_test_case(response.choices[0].message.content),
            }

        elif isinstance(client, anthropic.Anthropic):
            # Anthropic's messages.create API
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,  # Anthropic uses max_tokens, not max_output_tokens
                "temperature": temperature,
                "messages": messages,
                "reasoning": reasoning,
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

        elif isinstance(client, GeminiClient):
            # GeminiClient's generate_content
            # Note: Gemini's prompt format for direct content generation is a string or list of parts.
            # We will pass the `max_tokens` and `temperature` directly to the `generate_content` method
            # as kwargs, which the GeminiClient will map to `generation_config`.

            gemini_response = client.generate_content(
                prompt,
                model_name=model,
                max_tokens=max_tokens,  # Mapped to max_output_tokens internally
                temperature=temperature,
                reasoning=reasoning,
            )

            # Extract details from the raw Gemini response object
            generated_content = ""
            if gemini_response.candidates:
                # Ensure the response has at least one candidate and its content is valid
                if (
                    gemini_response.candidates[0].content
                    and gemini_response.candidates[0].content.parts
                ):
                    # Concatenate text from all parts if there are multiple (e.g., for tool outputs)
                    generated_content = "".join(
                        part.text
                        for part in gemini_response.candidates[0].content.parts
                        if part.text
                    )

            # Usage metadata can be found in response.usage_metadata
            input_tokens = (
                gemini_response.usage_metadata.prompt_token_count
                if gemini_response.usage_metadata
                else 0
            )
            output_tokens = (
                gemini_response.usage_metadata.candidates_token_count
                if gemini_response.usage_metadata
                else 0
            )

            # Stop reason is in candidates[0].finish_reason
            stop_reason = (
                gemini_response.candidates[0].finish_reason
                if gemini_response.candidates
                else None
            )

            return {
                "success": True,
                "content": generated_content,
                "model": model,  # Gemini response object doesn't directly return model name in same field
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "stop_reason": stop_reason,
                "extract_content": extract_test_case(generated_content),
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


def run(args):
    print(
        {
            "input_file": args.input_file,
            "output_file": args.output_file,
            "model": args.model,
        }
    )
    # api_dict = None
    with open(args.api_file) as f:
        api_dict = json.load(f)

    if args.model.startswith("o3-mini") or "gpt" in args.model:
        api_key = api_dict["gpt"]
    elif "deepseek" in args.model:
        api_key = api_dict["deepseek"]
    elif "gemini" in args.model:
        api_key = api_dict["gemini"]
    elif "claude" in args.model:
        api_key = api_dict["claude"]

    client = init_api(model=args.model, api_key=api_key)

    # read data: the json file from the input_file
    input_data = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            input_data.append(json.loads(line))

    result_dict = {}

    # process each data point to query:
    for inputTemp in input_data:
        uuid = inputTemp["uuid"]
        module = inputTemp["code_src"]
        branch = inputTemp["branches"]
        for item in branch:
            key = KEY_TEMPLATE.format(uuid, item)
            print(f"Key: {key}")
            execution_branch = ""
            test = branch[item]
            for t_branch in test:
                item_str = '->'.join(str(x) for x in t_branch)
                execution_branch += f"{item_str}\n"
    
            prompt = PROMPT_COT.format(module=module, execution_branch=execution_branch)
            response = query_prompt(
                prompt=prompt,
                client=client,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                reasoning=args.reasoning,
            )
            result_dict[key] = response

            # write the result_dict to the output file
            with open(args.output_file, "w") as f:
                json.dump(result_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some prompts.")
    parser.add_argument("--input_file", type=str, help="Path to the input file")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument(
        "--api_file", type=str, help="API File"
    )
    parser.add_argument(
        "--model", type=str, default="o3-mini-2025-01-31", help="Model to use for generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=1, help="Temperature for generation"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="Maximum tokens for generation"
    )
    parser.add_argument(
        "--reasoning", type=str, default="medium", help="Reasoning Effort"
    )
    args = parser.parse_args()
    run(args)

import openai
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "key": "bold yellow",
        "value": "white",
    }
)

console = Console(theme=custom_theme)


def chat_with_gpt(prompt, model="gpt-4o", temperature=0.7, max_tokens=200):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"].strip()

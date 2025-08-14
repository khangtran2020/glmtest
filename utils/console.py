from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from typing import Dict

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


def log_table(dct: Dict, name: str):

    table = Table(title=name)

    table.add_column("Property.", style="key")
    table.add_column("Values", style="value")

    for key in dct.keys():
        table.add_row(key, dct[key])
    console.log(table)

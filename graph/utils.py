import re
import sys
from graph.joerngraph import JoernGraph

# typing
from typing import List
from graph.core import Graph
from argparse import Namespace
from rich.console import Console


def get_graph(args: Namespace, graph_type: str, logger: Console) -> Graph:
    if graph_type == "joern":
        graph = JoernGraph(
            port=args.joern_port,
            joern_path=args.joern_path,
            logger=logger,
        )
    return graph

import re
import sys
from graph.joerngraph import JoernGraph

# typing
from typing import List


def get_graph(graph_type, args, logger):
    if graph_type == "joern":
        graph = JoernGraph(
            host=args.joern_host,
            port=args.joern_port,
            joern_path=args.joern_path,
            docker_image=args.joern_docker_image,
            logger=logger,
        )
    return graph

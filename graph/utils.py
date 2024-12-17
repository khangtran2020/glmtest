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
            graph_path=args.graph_path,
            docker_image=args.joern_docker_image,
            logger=logger,
        )
    return graph


def extract_list_content(input_string) -> List[str]:
    # Regular expression to find content between 'List(' and ')', including multiline content
    pattern = r"List\((.*?)\)"
    matches = re.findall(pattern, input_string, re.DOTALL)

    return matches


def handle_location_out(out_str: str) -> List[dict]:
    out_str = out_str.strip()
    lines = out_str.split("\n")[1:-1]  # Remove first and last line
    data = []
    stack = []
    for line in lines:
        stack.append(line)
        if "NewLocation(" in line:
            new_data = {}
            for key in ["filename", "lineNumber"]:
                new_data[key] = None  # line.split(key + " = ")[1].split(",")[0].strip()
        if (")" in line) & (len(line) - len(line.lstrip()) == 2):
            for l in stack:
                if "filename = " in line:
                    # print(line)
                    value = line.split("filename = ")[1].split(",")[0].strip()
                    new_data["filename"] = value if value != "<empty>" else None
                elif "lineNumber = " in line:
                    value = line.split("lineNumber = ")[1].split(",")[0].strip()
                    value = value if value != "None" else None
                    if value is not None:
                        value = int(value.split("value = ")[1].split(")")[0].strip())
                    new_data["lineNumber"] = value
                    break
            stack = []
            data.append(new_data)
    return data

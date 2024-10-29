import os
import ast
import networkx as nx
from copy import deepcopy
from networkx import DiGraph
from rich.console import Console
from typing import Tuple, Dict, Set, Union, List
from coverage.parser import PythonParser


def read_module(filepath: str, console: Console) -> str:

    console.log(f"Reading code from file: {filepath}")
    with open(filepath, "r") as f:
        code = f.read()
        num_line = len(code.split("\n"))

    return code, num_line


def analyze_code(code: str, console: Console) -> Tuple[Dict, Set, Set, Set]:

    # Parse the source code into an AST
    try:
        tree = ast.parse(code)
    except Exception as e:
        print("An error occur:", e)
        return -1

    # Initialize a list to store information
    analysis_results = {
        "others": {
            "start_modul": 1,
        },
        "functions": {},
        "async_functions": {},
        "classes": {},
    }
    for_line = []
    while_line = []
    key_start_line = []
    key_end_line = []

    # Walk through each node in the AST
    for node in ast.walk(tree):

        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", None)
            analysis_results["functions"][node.name] = (start_line, end_line)
            key_start_line.append(start_line)
            key_end_line.append(end_line)

        elif isinstance(node, ast.AsyncFunctionDef):
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", None)
            analysis_results["functions"][node.name] = (start_line, end_line)
            key_start_line.append(start_line)
            key_end_line.append(end_line)

        elif isinstance(node, ast.ClassDef):
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", None)
            analysis_results["functions"][node.name] = (start_line, end_line)
            key_start_line.append(start_line)
            key_end_line.append(end_line)

        elif isinstance(node, ast.For):
            for_line.append(node.lineno)

        elif isinstance(node, ast.While):
            while_line.append(node.lineno)

    log_info = (
        f"# start lines: {len(set(key_start_line))} "
        + f"# end lines: {len(set(key_end_line))} "
        + f"# loop lines: {len(set(for_line) + set(while_line))}"
    )
    console.log(log_info)

    return (
        analysis_results,
        set(key_start_line),
        set(key_end_line),
        set(for_line) + set(while_line),
    )


def parse_code(code: str, start_lines: Set, console: Console) -> DiGraph:

    parser = PythonParser(filename=None, exclude=None, text=code)
    parser.parse_source()
    arcs = parser.arcs()

    G = nx.DiGraph()
    for i in parser.statements:
        G.add_node(i)

    for e in arcs:
        if e[1] > 0:
            if e[1] not in start_lines:
                G.add_edge(*e)
        else:
            G.add_edge(e[0], 10000 + e[1] * -1)

    log_info = f"# arcs: {len(arcs)}"
    console.log(log_info)


def DFS_branch(
    G: DiGraph,
    node: int,
    visited_nodes: List = [],
    current_path: List = [],
    end_node: int = None,
    found_path: List = [],
    loop_list: Union[List, Set] = [],
) -> List:

    # Mark node as visited
    if node == end_node:
        found_path.append(deepcopy(current_path))
        print(found_path)

    visited_nodes[node] += 1
    current_path.append(node)

    # process something with current_node
    print("Processing current node:", node, end_node)

    for neighbor_node in G.neighbors(node):

        if visited_nodes[neighbor_node] == 0:
            DFS_branch(
                G=G,
                node=neighbor_node,
                current_path=current_path,
                visited_nodes=visited_nodes,
                end_node=end_node,
                found_path=found_path,
                loop_list=loop_list,
            )
        else:
            if (neighbor_node in loop_list) and (visited_nodes[neighbor_node] == 1):
                DFS_branch(
                    G=G,
                    node=neighbor_node,
                    current_path=current_path,
                    visited_nodes=visited_nodes,
                    end_node=end_node,
                    found_path=found_path,
                    loop_list=loop_list,
                )

    current_path.pop()
    visited_nodes[node] -= 1
    return found_path


def process_item(
    item_name: str,
    item_type: str,
    line_dict: Dict,
    loop_line: Union[Set, List],
    G: DiGraph,
    console: Console,
) -> List:
    start_line, end_line = line_dict[item_type][item_name]
    subloop = [
        line for line in loop_line if (line >= start_line) and (line <= end_line)
    ]

    statement = [
        node for node in G.nodes() if (node >= start_line) and (node <= end_line)
    ]

    statement.append(10000 + start_line)
    statement.sort()
    G_sub = G.subgraph(nodes=statement).copy()
    if G_sub.has_edge(start_line, statement[1]) == False:
        G_sub.add_edge(start_line, statement[1])

    visited = [0] * 20000
    found_path = DFS_branch(
        G=G_sub,
        node=start_line,
        visited_nodes=visited,
        current_path=[],
        end_node=10000 + start_line,
        found_path=[],
        loop_list=subloop,
    )

    log_info = f"Found: {len(found_path)} branches for {item_type}: {item_name}"
    console.log(log_info)
    return found_path


def process_module(filepath: str, console: Console):

    code, num_line = read_module(filepath=filepath, console=console)
    line_dict, start_line, end_line, loop_line = analyze_code(
        code=code, console=console
    )
    G = parse_code(code=code, start_lines=start_line, console=console)

    res_dict = {"func": {}, "async_func": {}, "class": {}}

    # process func
    console.log("Processing Function")
    item_type = "func"
    for func_name in line_dict["functions"].keys():
        found_branch = process_item(
            item_name=func_name,
            item_type=item_type,
            line_dict=line_dict,
            loop_line=loop_line,
            G=G,
            console=console,
        )
        res_dict["func"][func_name] = found_branch

    # process async func
    console.log("Processing Async Function")
    item_type = "async_func"
    for func_name in line_dict["async_functions"].keys():
        found_branch = process_item(
            item_name=func_name,
            item_type=item_type,
            line_dict=line_dict,
            loop_line=loop_line,
            G=G,
            console=console,
        )
        res_dict["async_func"][func_name] = found_branch

    # process class
    console.log("Processing Classes")
    item_type = "class"
    for func_name in line_dict["classes"].keys():
        found_branch = process_item(
            item_name=func_name,
            item_type=item_type,
            line_dict=line_dict,
            loop_line=loop_line,
            G=G,
            console=console,
        )
        res_dict["class"][func_name] = found_branch
    return res_dict

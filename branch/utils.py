import re
import os
import ast
import networkx as nx
from rich import print as pprint
from copy import deepcopy
from networkx import DiGraph
from rich.console import Console
from typing import Tuple, Dict, Set, Union, List
from coverage.parser import PythonParser
from coverage.data import CoverageData
from utils.utils import run_command

COVERAGE_TEMPLATE = """docker run --rm -v {}:/project -v {}:/test -v {}:/output -v {}:/package {} {} {}"""

NODE_CONSTANT = 1e6


def read_module(filepath: str) -> str:

    # console.log(f"Reading code from file: {filepath}")
    with open(filepath, "r") as f:
        code = f.read()
        num_line = len(code.split("\n"))

    return code


def analyze_code(code: str) -> Dict:

    # Parse the source code into an AST
    try:
        tree = ast.parse(code)
    except Exception as e:
        raise ValueError(f"Error parsing code: {e}")

    # Initialize a list to store information
    analysis_results = {
        "others": {
            "start_modul": 1,
        },
        "functions": {},
        "async_functions": {},
    }
    for_line = []
    while_line = []
    key_start_line = []
    key_end_line = []

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
            analysis_results["async_functions"][node.name] = (start_line, end_line)
            key_start_line.append(start_line)
            key_end_line.append(end_line)

        # elif isinstance(node, ast.ClassDef):
        #     start_line = node.lineno
        #     end_line = getattr(node, "end_lineno", None)
        #     analysis_results["classes"][node.name] = (start_line, end_line)
        #     key_start_line.append(start_line)
        #     key_end_line.append(end_line)

        elif isinstance(node, ast.For):
            for_line.append(node.lineno)

        elif isinstance(node, ast.While):
            while_line.append(node.lineno)

    return analysis_results


def parse_code(code: str) -> DiGraph:

    parser = PythonParser(filename=None, exclude=None, text=code)
    parser.parse_source()
    arcs = parser.arcs()

    set_of_statements = []
    set_of_startlines = []
    set_of_endlines = []

    for e in arcs:
        if e[0] > 0:
            set_of_statements.append(e[0])
        if e[1] > 0:
            set_of_statements.append(e[1])
        if e[1] < 0:
            if "import " not in code.split("\n")[-1 * e[1] - 1]:
                # print(e[1])
                set_of_startlines.append(-1 * e[1])
            set_of_endlines.append(e[0])

    set_of_statements = set(set_of_statements)
    set_of_startlines = set(set_of_startlines)
    set_of_endlines = set(set_of_endlines)

    # print(set_of_endlines)

    G = nx.DiGraph()
    for i in parser.statements:
        G.add_node(i)

    for e in arcs:
        if (e[1] > 0) and (e[1] not in set_of_startlines):
            line = code.split("\n")[e[1] - 1]
            if len(line) != len(line.lstrip()):
                G.add_edge(*e)

    # merge with the first statement
    sorted_statements = sorted(list(set_of_statements))
    for i in set_of_startlines:
        for j in sorted_statements:
            if j > i:
                G.add_edge(i, j)
                break

    # handle try except
    # Find try and except blocks
    tree = ast.parse(code)
    try_blocks = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            block_info = {"try": node.lineno, "excepts": []}
            for handler in node.handlers:
                end_line = getattr(handler, "end_lineno", None)
                block_info["excepts"].append((handler.lineno, end_line))
            try_blocks.append(block_info)

    # Add edges from each line between try and the first except to except blocks
    for block in try_blocks:
        try_start = block["try"]
        except_starts = [ex[0] for ex in block["excepts"]]

        # Find the first except that comes after the try
        first_except = None
        for ex_start in except_starts:
            if ex_start > try_start:
                first_except = ex_start
                break

        if first_except is not None:
            for line in sorted_statements:
                if try_start < line < first_except:
                    for ex_start, _ in block["excepts"]:
                        G.add_edge(line, ex_start)
    return G, sorted(list(set_of_endlines))


def DFS_branch(
    G: DiGraph,
    node: int,
    end_nodes: List[int],
    current_path: List[int],
    visited_nodes: List[int],
):

    current_path.append(node)
    visited_nodes.append(node)

    if node in end_nodes:
        yield current_path + [node]
    else:
        for neighbor in G.neighbors(node):
            # check if this neighbor is the start of a block
            if neighbor not in visited_nodes:
                yield from DFS_branch(
                    G=G,
                    node=neighbor,
                    end_nodes=end_nodes,
                    current_path=current_path,
                    visited_nodes=visited_nodes,
                )

    current_path.pop()
    visited_nodes.pop()


def get_all_branch(
    code: str = None, filepath: str = None, console: Console = None
) -> Dict:

    pprint("[green]Extracting branches...[/green]")

    if code is None and filepath is None:
        raise ValueError("Either code or filepath must be provided.")

    if code is None and filepath is not None:
        code = read_module(filepath=filepath)

    line_dict = analyze_code(code=code)
    G, set_of_endline = parse_code(code=code)

    branches = []
    num_branch = 0
    # process func
    if console is not None:
        console.log("Processing Function")
    for func_name in line_dict["functions"].keys():
        set_of_end = [
            e
            for e in set_of_endline
            if e > line_dict["functions"][func_name][0]
            and e <= line_dict["functions"][func_name][1]
        ]
        if len(set_of_end) == 0:
            continue
        for branch in DFS_branch(
            G=G,
            node=line_dict["functions"][func_name][0],
            end_nodes=set_of_end,
            current_path=[],
            visited_nodes=[],
        ):
            if branch[:-1] not in branches:
                num_branch += 1
                pprint(f"[blue]Found branch: {num_branch} - {len(branch)} [/blue]")
                branches.append(branch[:-1])  # remove the end node

    # process async func
    if console is not None:
        console.log("Processing Async Function")
    for func_name in line_dict["async_functions"].keys():
        set_of_end = [
            e
            for e in set_of_endline
            if e > line_dict["functions"][func_name][0]
            and e <= line_dict["functions"][func_name][1]
        ]
        if len(set_of_end) == 0:
            continue
        for branch in DFS_branch(
            G=G,
            node=line_dict["async_functions"][func_name][0],
            end_nodes=set_of_end,
            current_path=[],
            visited_nodes=[],
        ):
            if branch[:-1] not in branches:
                num_branch += 1
                pprint(f"[blue]Found branch: {num_branch} - {len(branch)}[/blue]")
                branches.append(branch[:-1])  # remove the end node

    return branches


def run_coverage(
    code_path: str,
    test_path: str,
    output_path: str,
    package_path: str,
    image_name: str,
    test_file: str,
    file_name: str,
    data_name: str = ".coverage",
) -> Union[List, None]:

    # create command
    command = COVERAGE_TEMPLATE.format(
        os.path.abspath(code_path),
        os.path.abspath(test_path),
        os.path.abspath(output_path),
        os.path.abspath(package_path),
        image_name,
        test_file,
        data_name,
    )
    print("Running command:", command)
    run_command(command, capture_output=False)
    data = CoverageData(
        basename=os.path.join(os.path.abspath(output_path), data_name),
        suffix=None,
        warn=None,
        debug=None,
    )
    data.read()
    arcs = data.arcs(filename=file_name)
    if arcs is None:
        return None
    # for e in arcs:
    branches = []
    visited = []
    for e in arcs:
        if e[0] < 0:
            continue
        if e[1] < 0:
            continue
        if e[0] in visited:
            for i, branch in enumerate(branches):
                if e[0] in branch:
                    branches[i].append(e[1])
                    visited.append(e[1])
        else:
            branches.append([e[0], e[1]])
            visited.append(e[0])
            visited.append(e[1])

    return branches


def merge_testcases(codes: List[str]) -> str:
    imports = merge_imports(codes=codes)
    function_list = []


def extract_imports_from_code(code):
    imports = set()
    for line in code.split("\n"):
        # Match 'import ...' or 'from ... import ...'
        if re.match(r"^\s*(import|from)\s+", line):
            imports.add(line.strip())
    return imports


def merge_imports(codes: List[str]):
    all_imports = set()
    for code in codes:
        all_imports.update(extract_imports_from_code(code=code))
    sorted_imports = sorted(all_imports)
    return "\n".join(sorted_imports)

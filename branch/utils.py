import re
import os
import ast
import sys
import networkx as nx
from rich import print as pprint
from rich.pretty import pretty_repr
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
    return G, sorted(list(set_of_endlines)), arcs


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


def is_name_main_check(node):
    """Check if node is: if __name__ == "__main__": or similar"""
    if not isinstance(node, ast.If):
        return False

    test = node.test

    # Check for: __name__ == "__main__"
    if isinstance(test, ast.Compare):
        if (
            isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            return True

        # Check for: "__main__" == __name__
        if (
            isinstance(test.left, ast.Constant)
            and test.left.value == "__main__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Name)
            and test.comparators[0].id == "__name__"
        ):
            return True

    return False


def get_init_lines(source_code: str):
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        return []

    num_lines = len(source_code.split("\n"))
    executed_line_numbers = set()

    for node in tree.body:
        # Skip if __name__ == "__main__" blocks
        if is_name_main_check(node):
            continue

        # Skip function and class definitions entirely (including decorators)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        # Everything else at top level executes
        if node.lineno and node.end_lineno:
            for line_num in range(node.lineno, node.end_lineno + 1):
                executed_line_numbers.add(line_num)

    # Return sorted list of (line_number, line_content)
    result = []
    for line_num in sorted(executed_line_numbers):
        if line_num <= num_lines:
            result.append(line_num)

    return result


def get_all_branch(
    code: str = None,
    filepath: str = None,
    console: Console = None,
    batch_size: int = 10,
    branch_limit: int = 1000,
) -> Dict:

    # pprint("[green]Extracting branches...[/green]")
    # pprint("-------------------------")
    # pprint(f"[blue]Processing code:[/blue]\n {code}")

    if code is None and filepath is None:
        raise ValueError("Either code or filepath must be provided.")

    if code is None and filepath is not None:
        code = read_module(filepath=filepath)

    line_dict = analyze_code(code=code)

    # Debugging
    # pprint(f"[green]Analyzed line dict:[/green] {pretty_repr(line_dict)}")

    new_code = []
    for i, line in enumerate(code.split("\n")):
        new_code.append(f"{i+1}: {line}")
    new_code = "\n".join(new_code)

    # pprint(f"[green]Source code:[/green]\n{new_code}")

    G, set_of_endline, arcs = parse_code(code=code)

    # Debugging
    # pprint(f"[green]Parsed arcs:[/green] {pretty_repr(arcs)}")
    # pprint(f"[green]Set of end lines:[/green] {pretty_repr(set_of_endline)}")
    # pprint(f"[green]Parsed graph nodes:[/green] {pretty_repr(G.nodes())}")
    # pprint(f"[green]Parsed graph edges:[/green] {pretty_repr(G.edges())}")

    init_lines = get_init_lines(source_code=code)

    init_branch = []
    init_arcs = []
    for i, arc in enumerate(arcs):
        if arc[0] in init_lines and arc[1] in init_lines:
            init_arcs.append(arc)

    for arc in init_arcs:
        if arc[0] not in init_branch:
            init_branch.append(arc[0])
        if arc[1] not in init_branch:
            init_branch.append(arc[1])

    init_branch = sorted(init_branch)

    branches = []
    num_branch = 0
    # process func
    if console is not None:
        console.log("Processing Function")

    for func_name in line_dict["functions"].keys():

        batch_branches = []
        branch_idx = 0

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
            if branch[:-1] not in batch_branches:
                num_branch += 1
                branch_idx += 1
                batch_branches.append(branch[:-1])  # remove the end node
                if branch_idx >= batch_size:
                    batch_branches = [init_branch] + batch_branches
                    branches.append(batch_branches)
                    batch_branches = []
                    branch_idx = 0

            if num_branch >= branch_limit:
                break

        if len(batch_branches) > 0:
            batch_branches = [init_branch] + batch_branches
            branches.append(batch_branches)

    # Debugging
    # pprint(f"[green]Total branches found so far:[/green] {num_branch}")
    # pprint(f"[green]Branches so far:[/green] {pretty_repr(branches)}")
    # sys.exit(0)

    # process async func
    if console is not None:
        console.log("Processing Async Function")

    for func_name in line_dict["async_functions"].keys():

        batch_branches = []
        branch_idx = 0

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
            if branch[:-1] not in batch_branches:
                num_branch += 1
                branch_idx += 1
                batch_branches.append(branch[:-1])  # remove the end node
                if branch_idx >= batch_size:
                    batch_branches = [init_branch] + batch_branches
                    branches.append(batch_branches)
                    batch_branches = []
                    branch_idx = 0

            if num_branch >= branch_limit:
                break

        if len(batch_branches) > 0:
            batch_branches = [init_branch] + batch_branches
            branches.append(batch_branches)

    pprint(
        f"[green]For file {filepath}: total branches found {num_branch} seperated to {len(branches)} batches[/green]"
    )

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
    imports = merge_imports_ast(codes=codes)
    body_code = ""

    for code in codes:
        body_code += remove_imports(code=code) + "\n\n"

    merged_code = imports + "\n\n" + body_code
    return merged_code


def remove_imports(code: str) -> str:
    try:
        tree = ast.parse(code)
    except Exception as e:
        return code  # Return original code if parsing fails

    # Remove Import and ImportFrom nodes from the top-level body
    new_body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    tree.body = new_body

    # Unparse the modified AST back to code (Python 3.9+)
    return ast.unparse(tree)


def extract_imports_with_ast(code: str):
    imports = set()
    import_from_dict = {}
    try:
        tree = ast.parse(code)
    except Exception as e:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    imports.add(f"import {alias.name} as {alias.asname}")
                else:
                    imports.add(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            # Handle relative imports
            if node.module is None and node.level > 0:
                module_prefix = "." * node.level
            else:
                module_prefix = "." * node.level + (node.module or "")

            if module_prefix not in import_from_dict.keys():
                import_from_dict[module_prefix] = []

            for alias in node.names:
                if alias.asname:
                    imports.add(
                        f"from {module_prefix} import {alias.name} as {alias.asname}"
                    )
                else:
                    import_from_dict[module_prefix].append(alias.name)
    for module_prefix in import_from_dict.keys():
        list_of_import = import_from_dict[module_prefix]
        import_line = ", ".join(list_of_import)
        imports.add(f"from {module_prefix} import {import_line}")
    return imports


def merge_imports_ast(codes: list):
    all_imports = set()
    for code in codes:
        all_imports.update(extract_imports_with_ast(code))
    sorted_imports = sorted(all_imports)
    return "\n".join(sorted_imports)


# def extract_imports_from_code(code):
#     imports = set()
#     for line in code.split("\n"):
#         # Match 'import ...' or 'from ... import ...'
#         if re.match(r"^\s*(import|from)\s+", line):
#             imports.add(line.strip())
#     return imports


# def merge_imports(codes: List[str]):
#     all_imports = set()
#     for code in codes:
#         all_imports.update(extract_imports_from_code(code=code))
#     sorted_imports = sorted(all_imports)
#     return "\n".join(sorted_imports)


def extract_functions_from_code(code: str) -> Dict[str, str]:
    functions = {}
    try:
        tree = ast.parse(code)
    except Exception as e:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", None)
            function_code = "\n".join(code.split("\n")[start_line - 1 : end_line])
            functions[node.name] = function_code
    return functions


def change_function_name(code: str, old_name: str, new_name: str) -> str:

    try:
        tree = ast.parse(code)
    except Exception as e:
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == old_name:
            node.name = new_name
            break

    ast.fix_missing_locations(node)
    # Unparse just this function node
    return ast.unparse(node)

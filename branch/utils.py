import re
import os
import ast
import networkx as nx
from rich import print as pprint
from copy import deepcopy
from networkx import DiGraph
from rich.console import Console
from dataclasses import dataclass
from typing import Tuple, Dict, Set, Union, List, Optional, Any
from coverage.parser import PythonParser
from coverage.data import CoverageData
from utils.utils import run_command

COVERAGE_TEMPLATE = """docker run --rm -v {}:/project -v {}:/test -v {}:/output -v {}:/package {} {} {}"""

NODE_CONSTANT = 1e6


@dataclass
class BlockSpan:
    kind: str  # "class" | "function" | "async_function"
    qualname: str  # e.g., "MyClass.method(x, y=?)", "helper(a)", "Outer.Inner"
    start: int  # inclusive (1-based)
    end: int  # inclusive (1-based)
    depth: int  # number of name components


def _node_span(node: ast.AST) -> Tuple[int, int]:
    """Compute [start, end] lines (1-based, inclusive) for class/def/async def nodes.
    Includes decorator lines if present. Requires Python 3.8+ for end_lineno.
    """
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)

    # Include decorators (if any)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        for dec in getattr(node, "decorator_list", []):
            dln = getattr(dec, "lineno", None)
            if isinstance(dln, int):
                start = min(start, dln) if isinstance(start, int) else dln

    # Fallback if end_lineno missing
    if not isinstance(end, int):
        end = int(start) if isinstance(start, int) else 1
        for child in ast.walk(node):
            e = getattr(child, "end_lineno", None)
            if isinstance(e, int):
                end = max(end, e)

    return int(start), int(end)


def _arg_name(a: ast.arg) -> str:
    # Just the identifier; we ignore annotations in the signature key
    return a.arg


def _normalize_signature(args: ast.arguments) -> str:
    """Return a normalized signature string that distinguishes argument shapes.

    - Shows positional-only params and "/" if any (PEP 570)
    - Shows var-positional as "*name" if present, else lone "*" before kw-only (PEP 3102)
    - Shows kw-only params (after "*" marker or *var)
    - Shows var-keyword as "**name"
    - Marks presence of a default with '=?' (without revealing default value)
    """
    parts: List[str] = []

    # Positional-only
    posonly = getattr(args, "posonlyargs", [])
    for a in posonly:
        parts.append(_arg_name(a))
    if posonly:
        parts.append("/")  # marker after the last pos-only

    # Positional-or-keyword (args.args) with defaults
    # Defaults align to the last N of args.args
    defaults = list(args.defaults or [])
    n_args = len(args.args)
    n_def = len(defaults)
    def_start = n_args - n_def
    for i, a in enumerate(args.args):
        name = _arg_name(a)
        if i >= def_start:
            parts.append(f"{name}=?")
        else:
            parts.append(name)

    # Var-positional
    if args.vararg is not None:
        parts.append(f"*{_arg_name(args.vararg)}")
        star_already = True
    else:
        star_already = False

    # Kw-only (with defaults in kw_defaults)
    for name, dflt in zip(args.kwonlyargs or [], args.kw_defaults or []):
        n = _arg_name(name)
        if dflt is None:
            parts.append(n)
        else:
            parts.append(f"{n}=?")

    # If there are kw-only args but no *vararg, we need a bare "*" marker in front
    if (args.kwonlyargs or []) and not star_already:
        # Insert "*" before the first kw-only item; find its index
        first_kw_idx = 0
        # find where kw-only params started (after pos params)
        # We placed posonly + "/" (optional) + args.args
        # So kw-only start index is len(parts) - len(kwonlyargs)
        first_kw_idx = len(parts) - len(args.kwonlyargs)
        parts.insert(first_kw_idx, "*")

    # Var-keyword
    if args.kwarg is not None:
        parts.append(f"**{_arg_name(args.kwarg)}")

    return "(" + ", ".join(parts) + ")"


def detect_loops(script_code: str):
    """
    Detects all 'for' and 'while' loops in the given Python script.

    Args:
        script_code (str): The Python source code as a string.

    Returns:
        dict: A dictionary with 'for' and 'while' as keys and lists of line numbers as values.
    """
    tree = ast.parse(script_code)
    for_lines = []
    while_lines = []

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            for_lines.append(node.lineno)
        elif isinstance(node, ast.While):
            while_lines.append(node.lineno)

    return {"for": sorted(for_lines), "while": sorted(while_lines)}


def _display_name(n: ast.AST) -> Optional[str]:
    """Return display name for a block:
    - class: its name
    - function/async function: name + normalized signature
    """
    if isinstance(n, ast.ClassDef):
        return n.name
    if isinstance(n, ast.FunctionDef):
        return f"{n.name}{_normalize_signature(n.args)}"
    if isinstance(n, ast.AsyncFunctionDef):
        return f"{n.name}{_normalize_signature(n.args)}"
    return None


def _walk_blocks(source: str) -> List[BlockSpan]:
    tree = ast.parse(source)
    blocks: List[BlockSpan] = []
    stack: List[str] = []

    def visit(n: ast.AST):
        disp = _display_name(n)
        kind: Optional[str] = None
        if isinstance(n, ast.ClassDef):
            kind = "class"
        elif isinstance(n, ast.FunctionDef):
            kind = "function"
        elif isinstance(n, ast.AsyncFunctionDef):
            kind = "async_function"

        if disp and kind:
            qual_parts = stack + [disp]
            qualname = ".".join(qual_parts)
            start, end = _node_span(n)
            blocks.append(BlockSpan(kind, qualname, start, end, len(qual_parts)))

            stack.append(disp)
            for child in ast.iter_child_nodes(n):
                visit(child)
            stack.pop()
        else:
            for child in ast.iter_child_nodes(n):
                visit(child)

    visit(tree)
    return blocks


def _find_loop_starts(source: str) -> Dict[int, str]:
    """Return {lineno: loop_type} where loop_type in {"for", "while"}.
    Includes AsyncFor as "for".
    """
    tree = ast.parse(source)
    loop_starts: Dict[int, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            ln = getattr(node, "lineno", None)
            if isinstance(ln, int):
                loop_starts[ln] = "for"
        elif isinstance(node, ast.While):
            ln = getattr(node, "lineno", None)
            if isinstance(ln, int):
                loop_starts[ln] = "while"
    return loop_starts


def classify_lines(source: str) -> Dict[int, Dict[str, object]]:
    """Return a dict keyed by line number (1-based)."""
    lines = source.splitlines()
    n = len(lines)

    # Initialize all as module
    result: Dict[int, Dict[str, object]] = {
        i
        + 1: {
            "block": "module",
            "kind": "module",
            "is_loop_start": False,
            "loop_type": None,
        }
        for i in range(n)
    }

    # Paint blocks (deeper overrides)
    blocks = _walk_blocks(source)
    blocks.sort(key=lambda b: (b.depth, b.start, -b.end))
    for b in blocks:
        for ln in range(max(1, b.start), min(n, b.end) + 1):
            cell = result[ln]
            cell["block"] = b.qualname
            cell["kind"] = b.kind

    # Mark loop starts
    loop_starts = _find_loop_starts(source)
    for ln, ltype in loop_starts.items():
        if 1 <= ln <= n:
            result[ln]["is_loop_start"] = True
            result[ln]["loop_type"] = ltype

    return result


def _end_lineno_fallback(node: ast.AST) -> int:
    """
    Best-effort fallback to compute an end line number if the node lacks
    the 'end_lineno' attribute (older Python or unusual nodes).
    We take the maximum end/line number of all descendants, or the node's
    own line number if nothing else is present.
    """
    best = getattr(node, "end_lineno", None)
    if isinstance(best, int):
        return best

    best = getattr(node, "lineno", 0)
    for child in ast.walk(node):
        end_ln = getattr(child, "end_lineno", None)
        ln = getattr(child, "lineno", None)
        if isinstance(end_ln, int):
            best = max(best, end_ln)
        elif isinstance(ln, int):
            best = max(best, ln)
    return best or 0


def _node_span(node: ast.AST) -> Tuple[int, int]:
    start = getattr(node, "lineno", None) or 0
    end = getattr(node, "end_lineno", None)
    if not isinstance(end, int):
        end = _end_lineno_fallback(node)
    return start, end


def _docstring_of(node: ast.AST) -> Optional[Tuple[int, int, str]]:
    """
    If 'node' (Module, ClassDef, FunctionDef/AsyncFunctionDef) has a docstring,
    return (start_line, end_line, text). Otherwise None.
    """
    body = getattr(node, "body", None)
    if not body or not isinstance(body, list) or not body:
        return None

    first = body[0]
    # In Python 3.8+, docstring appears as ast.Expr(value=ast.Constant(str))
    if isinstance(first, ast.Expr):
        val = first.value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            s, e = _node_span(first)
            return s, e, val.value
        # Older forms (rare today): ast.Str
        if hasattr(ast, "Str") and isinstance(val, ast.Str):  # type: ignore[attr-defined]
            s, e = _node_span(first)
            return s, e, val.s  # type: ignore[attr-defined]
    return None


def _extract_decorators(node: ast.AST, source: str) -> List[str]:
    """
    Extract decorator names/expressions from a function or class definition.
    Returns a list of decorator strings as they appear in the source.
    """
    decorators = []
    decorator_list = getattr(node, "decorator_list", [])

    for dec in decorator_list:
        # Try to extract a meaningful representation
        if isinstance(dec, ast.Name):
            decorators.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            # Handle chained attributes like @obj.method
            parts = []
            current = dec
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            decorators.append(".".join(reversed(parts)))
        elif isinstance(dec, ast.Call):
            # Decorator with arguments like @decorator(arg)
            func = dec.func
            if isinstance(func, ast.Name):
                decorators.append(f"{func.id}(...)")
            elif isinstance(func, ast.Attribute):
                parts = []
                current = func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                decorators.append(".".join(reversed(parts)) + "(...)")
            else:
                decorators.append("(...)")
        else:
            # Fallback for complex expressions
            decorators.append("<complex>")

    return decorators


def _qualname(name_stack: List[str], leaf: Optional[str]) -> Optional[str]:
    if leaf is None:
        return None
    parts = [*name_stack, leaf] if name_stack else [leaf]
    return ".".join(parts)


class SpanCollector(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.items: List[Dict[str, Any]] = []
        self.stack: List[str] = []  # for qualified names

    def record_entity(self, kind: str, name: Optional[str], node: ast.AST):
        start, end = _node_span(node)
        decorators = _extract_decorators(node, self.source)

        entity = {
            "kind": kind,  # "class" | "function" | "async_function"
            "name": _qualname(self.stack, name) if name else None,
            "start_line": start,
            "end_line": end,
        }

        # Only add decorators field if there are decorators
        if decorators:
            entity["decorators"] = decorators

        self.items.append(entity)

    def record_docstring(
        self, owner_kind: str, owner_name: Optional[str], s: int, e: int, text: str
    ):
        self.items.append(
            {
                "kind": "docstring",
                "of": owner_kind,  # "module" | "class" | "function"
                "name": _qualname(self.stack, owner_name) if owner_name else None,
                "start_line": s,
                "end_line": e,
                "text": text,
            }
        )

    # Module
    def visit_Module(self, node: ast.Module):
        ds = _docstring_of(node)
        if ds:
            s, e, text = ds
            self.record_docstring("module", None, s, e, text)
        self.generic_visit(node)

    # Class
    def visit_ClassDef(self, node: ast.ClassDef):
        self.record_entity("class", node.name, node)
        # push for qualname of nested members
        self.stack.append(node.name)
        # class docstring
        ds = _docstring_of(node)
        if ds:
            s, e, text = ds
            self.record_docstring("class", node.name, s, e, text)
        self.generic_visit(node)
        self.stack.pop()

    # Function (sync)
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.record_entity("function", node.name, node)
        # push for nested defs
        self.stack.append(node.name)
        ds = _docstring_of(node)
        if ds:
            s, e, text = ds
            self.record_docstring("function", node.name, s, e, text)
        self.generic_visit(node)
        self.stack.pop()

    # Function (async)
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.record_entity("async_function", node.name, node)
        self.stack.append(node.name)
        ds = _docstring_of(node)
        if ds:
            s, e, text = ds
            self.record_docstring("function", node.name, s, e, text)
        self.generic_visit(node)
        self.stack.pop()


def annotate_eliminate_lines(source_code: str) -> List[Dict[str, Any]]:
    tree = ast.parse(source_code, filename="<string>")
    collector = SpanCollector(source_code)
    collector.visit(tree)
    # Sort by start_line for a stable, readable output
    collector.items.sort(
        key=lambda x: (x["start_line"], x["end_line"], x.get("kind", ""))
    )
    return collector.items


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


def indent_text(text, indent_level):
    return "\n".join(
        " " * indent_level + line if line.strip() else line for line in text.split("\n")
    )


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
            "start_module": 1,
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

    # get all arcs
    parser = PythonParser(filename=None, exclude=None, text=code)
    parser.parse_source()
    arcs = parser.arcs()

    # cleaning arcs
    init_lines = get_init_lines(source_code=code)
    annotated_res = annotate_eliminate_lines(source_code=code)
    line_exclude = []
    for l in annotated_res:
        if l["kind"] == "docstring":
            line_exclude.append(l["start_line"])
            line_exclude.append(l["end_line"])
        elif l["kind"] == "class":
            line_exclude.append(l["start_line"])
        elif (l["kind"] == "function") or (l["kind"] == "async_function"):
            # print(l)
            if "decorators" in l.keys():
                i = 1
                for decor in l["decorators"]:
                    line_exclude.append(l["start_line"] - i)
                    i += 1
            line_exclude.append(l["start_line"])

    clean_arcs = []
    set_of_endlines = []
    for arc in arcs:
        if arc[1] in line_exclude:
            continue

        if arc[1] < 0:
            set_of_endlines.append(arc[0])

        if arc[0] < 0:
            if arc[1] == -1 * arc[0]:
                continue
            else:
                clean_arcs.append((-1 * arc[0], arc[1]))
        elif arc[0] in line_exclude:
            if arc[1] in line_exclude:
                continue
            elif arc[1] < 0:
                continue
            else:
                clean_arcs.append(arc)
        else:
            clean_arcs.append(arc)

    # Removing init arcs
    clean_arcs = sorted(clean_arcs)
    init_arcs = []
    remain_arcs = []
    for i, arc in enumerate(clean_arcs):
        if arc[0] in init_lines and arc[1] in init_lines:
            init_arcs.append(arc)
        else:
            remain_arcs.append(arc)

    G = nx.DiGraph()
    for i in parser.statements:
        G.add_node(i)

    for e in remain_arcs:
        if (e[1] > 0) and (e[1] not in line_exclude):
            line = code.split("\n")[e[1] - 1]
            if len(line) != len(line.lstrip()):
                G.add_edge(*e)

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
            for line in parser.statements:
                if try_start < line < first_except:
                    for ex_start, _ in block["excepts"]:
                        G.add_edge(line, ex_start)

    set_of_endlines = set(set_of_endlines)
    return G, init_arcs, set_of_endlines


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


def get_all_branch(
    code: str = None, num_branch_limit: int = 1000, batch_size: int = 10
) -> Dict:

    # pprint("[green]Extracting branches...[/green]")

    # Get all arcs & build graph of execution lines
    G, init_arcs, set_of_endlines = parse_code(code=code)

    branches = []
    # build init branch
    init_branch = []
    for arc in init_arcs:
        if arc[0] not in init_branch:
            init_branch.append(arc[0])
        if arc[1] not in init_branch:
            init_branch.append(arc[1])

    branches.append(init_branch)
    line_dict = analyze_code(code=code)

    num_branch = 1
    # process func

    for func_name in line_dict["functions"].keys():

        batch_branches = []
        set_of_end = [
            e
            for e in set_of_endlines
            if e > line_dict["functions"][func_name][0]
            and e <= line_dict["functions"][func_name][1]
        ]

        pprint(f"[blue]Function {func_name} has end lines: {set_of_end}[/blue]")
        if len(set_of_end) == 0:
            continue

        i = 0
        for branch in DFS_branch(
            G=G,
            node=line_dict["functions"][func_name][0],
            end_nodes=set_of_end,
            current_path=[],
            visited_nodes=[],
        ):
            if branch[:-1] not in batch_branches:
                num_branch += 1
                i += 1
                batch_branches.append(branch[:-1])  # remove the end node
                if i >= batch_size:
                    branches.append(batch_branches)
                    batch_branches = []
                    i = 0

            if num_branch >= num_branch_limit:
                break

        if len(batch_branches) > 0:
            branches.append(batch_branches)

    # process async func
    for func_name in line_dict["async_functions"].keys():

        batch_branches = []
        i = 0

        set_of_end = [
            e
            for e in set_of_endlines
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
                i += 1
                batch_branches.append(branch[:-1])  # remove the end node
                if i >= batch_size:
                    branches.append(batch_branches)
                    batch_branches = []
                    i = 0

            if num_branch >= num_branch_limit:
                break

        if len(batch_branches) > 0:
            branches.append(batch_branches)

    pprint(f"[blue]Total branches found: {num_branch}[/blue]")
    return branches

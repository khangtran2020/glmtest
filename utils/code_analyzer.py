import ast
import textwrap


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self, code):
        self.code = code.splitlines()
        self.imports = []
        self.functions = []
        self.classes = []
        self._analyze()

    def _get_code_block(self, node):
        return textwrap.dedent("\n".join(self.code[node.lineno - 1 : node.end_lineno]))

    def visit_Import(self, node):
        self.imports.append(("import", [alias.name for alias in node.names]))

    def visit_ImportFrom(self, node):
        self.imports.append(
            (f"from {node.module}", [alias.name for alias in node.names])
        )

    def visit_FunctionDef(self, node):
        function_info = {
            "name": node.name,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "code": self._get_code_block(node),
        }
        self.functions.append(function_info)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        class_info = {
            "name": node.name,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "methods": [],
            "code": self._get_code_block(node),
        }
        for subnode in node.body:
            if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = {
                    "name": subnode.name,
                    "start_line": subnode.lineno,
                    "end_line": subnode.end_lineno,
                    "code": self._get_code_block(subnode),
                }
                class_info["methods"].append(method_info)
        self.classes.append(class_info)

    def _analyze(self):
        tree = ast.parse("\n".join(self.code))
        self.visit(tree)

    def get_results(self):
        return {
            "imports": self.imports,
            "functions": self.functions,
            "classes": self.classes,
        }


# Example Usage
def analyze_code(code_snippet):
    try:
        analyzer = CodeAnalyzer(code_snippet)
        return analyzer.get_results()
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        return None


def remove_method_from_class(code, class_name, method_name):
    """Removes a method from a class in the given Python source code."""
    lines = code.splitlines()
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        print(code)
        return code
    new_code_lines = lines[:]

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for subnode in node.body:
                if (
                    isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and subnode.name == method_name
                ):
                    # Remove the method lines
                    for i in range(subnode.lineno - 1, subnode.end_lineno):
                        new_code_lines[i] = ""

    # Rebuild the source code without the method
    cleaned_code = "\n".join([line for line in new_code_lines])
    return cleaned_code

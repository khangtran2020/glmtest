import os
import ast
from tqdm import tqdm
from rich.progress import Progress
from rich.console import Console
from graph.core import Graph
from utils.utils import run_command

# typing
from typing import List, Union

PYNGUIN_TEMPLATE = """docker run --rm -v {}:/input:ro -v {}:/output -v {}:/package:ro {} \
    --module-name {} --coverage_metrics BRANCH --maximum_search_time {} --report-dir /output --project_path /input --output-path /output --output_variables TargetModule,CoverageTimeline --assertion-generation NONE"""


class Data(object):

    def __init__(
        self,
        name: str,
        path: str,
        logger: Console,
        graph: Graph,
        num_cpu: int,
        debug: bool = False,
    ) -> None:
        self.name = name  # name of the data
        self.path = path  # path of the raw data
        self.logger = logger
        self.graph = graph
        self.num_cpu = num_cpu

    def crawl(self) -> None:
        """
        Crawl the projects to the given path
        """
        pass

    def process_raw(self) -> None:
        """
        - process the raw data to extract the modules and functions
        - create a self.data object to store the extracted data
        - save them to the given path in json format
        """
        pass

    def create_module_info(self) -> List[dict]:
        """
        Create a module info from the extracted data
        Each module info icnludes:
            - module_name_test_gen (e.g., path.to.module, without .py)
            - module_path
            - project
            - project_path
            - code_path to raw project
            - output_test_path
            - package_path
            - module_name_after_test_gen
            - graph_path
            - graph_name
            - module_name_coverage
        """
        pass

    def process_test_gen(self) -> None:
        """
        - process the extracted data to generate test cases
        - save the test cases to the given path
        """
        # check if self.data is not None
        if self.data is None:
            self.logger.log("[red]No data to process[/red]")
            return

        # create module info from self.data
        module_info = self.create_module_info()
        if self.debug:
            module_info = module_info[:1]

        # process each module
        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Processing test generation for {self.name}[/cyan]",
                total=len(module_info),
            )
            for module in module_info:
                self.process_one_module(module)
                progress.update(task, advance=1)

    def process_one_module(self, module_info) -> List[dict]:

        module_results_info = {}

        # gen test case with pynguin
        command = self.get_pynguin_command_for_module(module_info)
        run_command(command=command, capture_output=False)
        # check if test case is generated
        if not os.path.exists(
            os.path.join(
                module_info["output_test_path"],
                module_info["module_name_after_test_gen"],
            )
        ):
            self.logger.log(
                f"[red]Test case for {module_info['module_name']} is not generated[/red]"
            )
            return []

        module_results_info["module_name"] = module_info["module_name"]
        module_results_info["module_path"] = module_info["module_path"]
        module_results_info["project"] = module_info["project"]

        # extract joern graph & locations
        self.graph.extract_graph(
            module_info["module_path"],
            save_path=os.path.join(
                module_info["graph_path"], module_info["graph_name"]
            ),
        )

        # if test case is generated, store the test case
        # count number of test cases
        test_path = os.path.join(
            module_info["output_test_path"], module_info["module_name_after_test_gen"]
        )
        # check correct path
        assert os.path.exists(test_path)
        sub_test_path = os.path.join(
            module_info["output_test_path"], module_info["module_name"]
        )
        os.makedirs(
            sub_test_path,
            exist_ok=True,
        )
        self.extract_functions_with_imports(
            file_path=test_path, save_path=sub_test_path
        )

        # split test case into test cases
        # run each test case with coverage.py
        # get the data and analyze the data
        return []

    def get_pynguin_command_for_module(self, module_info: dict) -> str:

        pynguin_command = PYNGUIN_TEMPLATE.format(
            os.path.abspath(module_info["code_path"]),
            os.path.abspath(module_info["output_test_path"]),
            os.path.abspath(module_info["package_path"]),
            self.docker_image,
            module_info["module_name_test_gen"],
            self.run_time,
        )
        return pynguin_command

    def count_test_cases(self, test_file: str) -> int:

        try:
            with open(test_file, "r") as file:
                file_content = file.read()

            # Parse the file content into an AST
            tree = ast.parse(file_content)

            # Count the number of function definitions
            function_count = sum(
                isinstance(node, ast.FunctionDef) for node in ast.walk(tree)
            )
            return function_count

        except Exception as e:
            print(f"An error occurred: {e}")
            return 0

    def extract_functions_with_imports(
        self, file_path: str, save_path: str
    ) -> Union[None, int]:

        try:
            with open(file_path, "r") as file:
                file_content = file.read()

            # Parse the Python file into an AST
            tree = ast.parse(file_content)

            # Collect all imports and function definitions
            imports = []
            functions = []

            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(node)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node)

            # Convert imports to code strings
            import_code = "\n".join(ast.unparse(import_node) for import_node in imports)

            # Convert each function to a string
            function_dict = {}
            for func in functions:
                func_code = ast.unparse(func)
                function_name = func.name
                function_dict[function_name] = f"{import_code}\n\n{func_code}"

            for i, key in enumerate(function_dict.keys()):
                with open(os.path.join(save_path, f"test_case_{i}.py"), "w") as file:
                    file.write(function_dict[key])

        except Exception as e:
            print(f"An error occurred: {e}")
            return -1

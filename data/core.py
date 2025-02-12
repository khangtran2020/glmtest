import os
import ast
import json
import torch
import numpy as np
import pandas as pd
from rich.progress import Progress
from rich.console import Console
from graph.core import Graph
from transformers import PreTrainedModel, PreTrainedTokenizer
from branch.utils import run_coverage
from utils.utils import run_command
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy

# typing
from typing import List, Union, Dict

PYNGUIN_TEMPLATE = """docker run --rm -v {}:/input:ro -v {}:/output -v {}:/package:ro {} \
    --module-name {} --coverage_metrics BRANCH --maximum_search_time {} --report-dir /output --project_path /input --output-path /output --output_variables TargetModule,CoverageTimeline --assertion-generation NONE"""


class Data(object):
    """
    Data class to crawl, process and generate test cases for the given data
    In the end, it will save the test cases to the given path in json format
    The data will be saved in the following format:
    {
        "uuid": unique id for each data point,
        "code_path": path to the code,
        "test_cases": {
            "test_case_1": {
                "test_case": test case 1,
                "branch": arcs for the test case 1
            },
            "test_case_2": {
                "test_case": test case 2,
                "branch": arcs for the test case 2
            }
            ...
        }
        "graph": {
            "src_graph_path": path to the graph of src_code,
            "node_feature_path": path to the node feature of src_code,
            "mask_path": path to the mask of branches,
        }
    }
    """

    def __init__(
        self,
        name: str,
        path: str,
        logger: Console,
        graph: Graph,
        num_cpu: int,
        feat_model: PreTrainedModel = None,
        feat_tokenizer: PreTrainedTokenizer = None,
        coverage_image: str = "coverage",
        debug: bool = False,
    ) -> None:
        self.name = name  # name of the data
        self.path = path  # path of the raw data
        self.logger = logger
        self.graph = graph
        self.num_cpu = num_cpu
        self.debug = debug
        self.feat_model = feat_model
        self.feat_tokenizer = feat_tokenizer
        self.coverage_image = coverage_image

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
        module_infos = self.create_module_info()
        if self.debug:
            module_infos = module_infos[:15]

        # process each module
        results = []
        for i, module_info in enumerate(module_infos):
            res = self.process_one_module(module_info)
            if res == {}:  # if no test case is generated
                continue
            res["uuid"] = i
            results.append(res)
            # save the processed data every 10 modules
            if len(results) % 10 == 0:
                with open(
                    os.path.join(self.data_path, "processed_data.json"), "w"
                ) as file:
                    json.dump(results, file, indent=4)
        self.processed_data = results
        #  save the processed data
        with open(os.path.join(self.data_path, "processed_data.json"), "w") as file:
            json.dump(results, file, indent=4)

    def process_one_module(self, module_info) -> dict:

        module_results_info = {}
        # check if test directory is created
        if not os.path.exists(module_info["output_test_path"]):
            os.makedirs(module_info["output_test_path"], exist_ok=True)

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
            return {}

        module_results_info["module_name_full"] = module_info["module_name_full"]
        module_results_info["module_path"] = module_info["module_path"]
        module_results_info["project"] = module_info["project"]
        module_results_info["test_cases"] = {}

        # extract joern graph & locations
        # check if path for graph exists
        if not os.path.exists(module_info["graph_path"]):
            os.makedirs(module_info["graph_path"], exist_ok=True)
        self.graph.extract_graph(
            module_info["module_path"],
            save_path=os.path.join(
                module_info["graph_path"], module_info["graph_name"]
            ),
        )
        module_results_info["graph_path"] = os.path.join(
            module_info["graph_path"], module_info["graph_name"]
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
        os.umask(0)
        os.makedirs(
            sub_test_path,
            exist_ok=True,
            mode=0o777,
        )
        self.extract_functions_with_imports(
            file_path=test_path, save_path=sub_test_path
        )

        if self.debug:
            self.logger.log(
                f"[green]Test case for {module_info['module_name']} is generated[/green]"
            )
            self.logger.log(f"Test cases are saved in {sub_test_path}")
            self.logger.log(f"Module file is: {module_info['module_name_coverage']}")
        # run each test case with coverage.py
        for i, test_file in enumerate(os.listdir(sub_test_path)):
            arcs = run_coverage(
                code_path=module_info["code_path"],
                test_path=sub_test_path,
                output_path=sub_test_path,
                package_path=module_info["package_path"],
                image_name=self.coverage_image,
                test_file=test_file,
                file_name=f"/project/{module_info['module_name_coverage']}.py",
            )
            if arcs is None:
                continue
            module_results_info["test_cases"][f"test_case_{i}"] = {}
            module_results_info["test_cases"][f"test_case_{i}"]["test_path"] = (
                os.path.join(sub_test_path, test_file)
            )
            module_results_info["test_cases"][f"test_case_{i}"]["branch"] = arcs
            if self.debug:
                break
        # get the data and analyze the data
        return module_results_info

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

    def get_mask_tensor(self, graph: Dict, branch: List) -> torch.Tensor:

        mask = np.zeros(len(graph["nodes"]))
        line_list = list(set(np.concatenate(np.array(branch, dtype=object)).tolist()))
        for i in range(len(graph["nodes"])):
            node = graph["nodes"][i]
            if node["location"]["filename"] == "N/A":
                try:
                    if node["properties"]["LINE_NUMBER"] in line_list:
                        mask[i] = 1
                except:
                    mask[i] = 0
        mask = torch.Tensor([mask])
        self.logger.log(f"Size of mask: {mask.size()}")
        return mask

    def get_node_features(self, graph: Dict) -> torch.Tensor:
        df = self.preprocess(graph)
        embeddings = []
        labels = df["LABELS"]

        # Encode LABELS to integers
        label_encoder = LabelEncoder()
        df["LABELS_ENCODED"] = label_encoder.fit_transform(labels)

        # Get Code Embedding
        for code in df["CODE"].tolist():
            inputs = self.tokenizer(
                code, padding=True, truncation=True, return_tensors="pt", max_length=128
            ).to(self.model.device)
            with torch.no_grad():
                embedding = self.model.encoder(**inputs).last_hidden_state.mean(dim=1)[
                    0
                ]
            embeddings.append(embedding.to("cpu").numpy())

        df["CODE_FEATURE"] = embeddings
        # df = df.drop(["LABELS","CODE"],axis=1)
        c_features = deepcopy(df["CODE_FEATURE"])
        df = df[["LABELS_ENCODED", "COLUMN_NUMBER", "ORDER", "LINE_NUMBER"]]
        feat_df = torch.from_numpy(df.values).float()
        c_features = np.concatenate([np.expand_dims(e, 0) for e in c_features], axis=0)
        c_features = torch.from_numpy(c_features).float()
        feat = torch.cat([feat_df, c_features], dim=1)
        self.logger.log(f"Size of node features: {feat.size()}")
        return feat

    def preprocess(self, graph):
        labels = []
        cnum = []
        order = []
        code = []
        lnum = []

        for node in graph["nodes"]:
            properties = node["properties"]
            labels.append(node["label"])
            try:
                cnum.append(properties["COLUMN_NUMBER"])
            except:
                cnum.append(-1)
            try:
                order.append(properties["ORDER"])
            except:
                order.append(-1)

            try:
                lnum.append(properties["LINE_NUMBER"])
            except:
                lnum.append(-1)

            try:
                if properties["CODE"] != "":
                    code.append(properties["CODE"])
                else:
                    code.append("EMPTY")
            except:
                code.append("EMPTY")

        nodes = pd.DataFrame(
            {
                "LABELS": np.array(labels),
                "COLUMN_NUMBER": np.array(cnum),
                "ORDER": np.array(order),
                "CODE": np.array(code),
                "LINE_NUMBER": np.array(lnum),
            }
        )
        return nodes

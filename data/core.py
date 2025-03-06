import os
import ast
import dgl
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rich.progress import Progress
from rich.console import Console
from graph.core import Graph
from transformers import PreTrainedModel, PreTrainedTokenizer
from branch.utils import run_coverage
from utils.utils import run_command, get_index_by_value
from utils.code_analyzer import analyze_code, remove_method_from_class
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy

# typing
from typing import List, Union, Dict

PYNGUIN_TEMPLATE = """docker run --rm -v {}:/input:ro -v {}:/output -v {}:/package:ro {} \
    --module-name {} --coverage_metrics BRANCH --maximum_search_time {} --report-dir /output --project_path /input --output-path /output --output_variables TargetModule,CoverageTimeline --assertion-generation NONE"""

PROMPT_CODE = """Generate the test case for the code below:
```python
{}
```
"""

PROMPT_GRAPH = """Generate the test case for the graph embedding of a targeted execution branch below:
{}
"""

PROMPT_CODE_GRAPH = """Generate the test case for the code below and the corresponding graph embedding of a targeted execution branch:

Here is the code:
```python
{}
```

Here is the graph embedding:
{}
"""

RESPONSE_TEMPLATE = """Here is the test case:
```python
{}
```
"""


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
        llm_tokenizer: PreTrainedTokenizer = None,
        coverage_image: str = "coverage",
        debug: bool = False,
        baseline_prompt: str = "code",
    ) -> None:
        self.name = name  # name of the data
        self.path = path  # path of the raw data
        self.logger = logger
        self.graph = graph
        self.num_cpu = num_cpu
        self.debug = debug
        self.feat_model = feat_model
        self.feat_tokenizer = feat_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.coverage_image = coverage_image
        self.baseline_prompt = baseline_prompt

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
        return mask

    def get_node_features(self, graph: Dict) -> torch.Tensor:
        df = self.preprocess(graph)
        embeddings = []
        labels = df["LABELS"]

        # Encode LABELS to integers
        label_encoder = LabelEncoder()
        df["LABELS_ENCODED"] = label_encoder.fit_transform(labels)

        # Get Code Embedding
        self.logger.log("[green]Embedding code...[/green]")
        for code in df["CODE"].tolist():
            inputs = self.feat_tokenizer(
                code,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128,
            ).to(self.feat_model.device)
            with torch.no_grad():
                embedding = self.feat_model.encoder(**inputs).last_hidden_state.mean(
                    dim=1
                )[0]
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

    def prepare_data(self) -> None:
        """
        Prepare the training data for the model
        """
        assert self.data is not None
        with self.logger.status("[green]Preparing data...[/green]"):
            self.processed_data = []
            for uuid, dat in self.data.items():
                with open(dat["code_path"], "r") as file:
                    src_code = file.read()
                graph = self.read_graph(dat)
                mask = torch.load(dat["graph"]["mask_path"], weights_only=True)
                num_tokens = []
                for testcase in dat["test_cases"].keys():
                    test_code = dat["test_cases"][testcase]["test_case"]
                    test_code = self.add_fuzz_tags(test_code)
                    if test_code == "N/A":
                        continue
                    mask_key = int(testcase.split("_")[-1])
                    branch = mask[mask_key]
                    branch_line = dat["test_cases"][testcase]["branch"]
                    active_node = get_index_by_value(a=branch, val=1)

                    prompt, response, full_text = self.get_prompt(
                        src_code=src_code,
                        testcase_out=test_code,
                        mask=active_node,
                        tokenizer=self.llm_tokenizer,
                        branch=branch_line,
                    )

                    num_token = len(self.llm_tokenizer.tokenize(full_text))
                    num_tokens.append(num_token)

                    graph_dict = {
                        key: graph[key]
                        for key in graph.keys()
                        if isinstance(graph[key], dgl.DGLGraph)
                    }
                    data = {
                        "uuid": uuid,
                        "prompt": prompt,
                        "response": response,
                        "full_text": full_text,
                        "graph": graph_dict,
                        "mask": mask[mask_key],
                    }
                    self.processed_data.append(data)
        self.logger.log("[green]Data is ready![/green]")
        self.logger.log(f"Size of data data: {len(self.processed_data)}")
        quartiles = np.quantile(num_tokens, [0, 0.25, 0.5, 0.75, 1])
        self.logger.log(f"Statistics of # tokens: {quartiles}")

    def read_graph(self, data: dict) -> dict:

        graph_path = data["graph"]["src_graph_path"]
        with open(graph_path, "r") as file:
            graph = json.load(file)

        graph_dict = {}
        num_nodes = len(graph["nodes"])
        feat = torch.load(data["graph"]["node_feature_path"], weights_only=True)
        assert num_nodes == feat.shape[0]

        edge_dict = self.read_edge(graph)

        for etype in edge_dict.keys():
            u = torch.Tensor(edge_dict[etype][0]).long()
            v = torch.Tensor(edge_dict[etype][1]).long()
            graph = dgl.graph((u, v), num_nodes=num_nodes)
            graph.ndata["feat"] = feat
            graph_dict[etype] = graph
        graph_dict["num_nodes"] = num_nodes
        graph_dict["feat_size"] = feat.size()
        return graph_dict

    def read_edge(self, graph: dict) -> dict:
        node_dict = self.get_node_id_dict(graph)
        edge_dict = {}
        for edge in graph["edges"]:
            if edge["label"] not in edge_dict:
                edge_dict[edge["label"]] = [
                    [node_dict[edge["src"]]],
                    [node_dict[edge["dst"]]],
                ]
            else:
                edge_dict[edge["label"]][0].append(node_dict[edge["src"]])
                edge_dict[edge["label"]][1].append(node_dict[edge["dst"]])
        return edge_dict

    def get_node_id_dict(self, graph: dict) -> dict:
        node_dict = {}
        for i in range(len(graph["nodes"])):
            node = graph["nodes"][i]
            node_dict[node["id"]] = i
        return node_dict

    def get_prompt(
        self,
        src_code: str,
        testcase_out: str,
        mask: torch.Tensor,
        branch: List,
        tokenizer: PreTrainedTokenizer,
    ):

        graph_pad = "<|graph_pad|>" * mask.size(0)
        if self.baseline_prompt == "code":
            text = PROMPT_CODE.format(src_code)
            response = RESPONSE_TEMPLATE.format(testcase_out)
        elif self.baseline_prompt == "graph":
            text = PROMPT_GRAPH.format(graph_pad)
            response = RESPONSE_TEMPLATE.format(testcase_out)
        elif self.baseline_prompt == "code_graph":
            text = PROMPT_CODE_GRAPH.format(src_code, graph_pad)
            response = RESPONSE_TEMPLATE.format(testcase_out)
        elif self.baseline_prompt == "code_tr":
            trucated_code = self.truncate_code(src_code=src_code, branch=branch)
            text = PROMPT_CODE.format(trucated_code)
            response = RESPONSE_TEMPLATE.format(testcase_out)
        elif self.baseline_prompt == "graph_tr":
            trucated_code = self.truncate_code(src_code=src_code, branch=branch)
            text = PROMPT_CODE_GRAPH.format(trucated_code, graph_pad)
            response = RESPONSE_TEMPLATE.format(testcase_out)

        task_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        )
        return text, response, task_prompt

    def add_fuzz_tags(self, code: str, tag: str = "fuzz") -> str:

        # print(code)
        try:
            tree = ast.parse(code)
        except:
            return "N/A"
        locations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                node_location = (
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset,
                )
                locations.append(node_location)
        locations = sorted(locations, key=lambda x: x[0])
        lines = code.split("\n")

        for loc in locations:
            start_line, start_col, end_line, end_col = loc
            if start_line == end_line:
                lines[start_line - 1] = (
                    lines[start_line - 1][:start_col]
                    + f"<|{tag}|>"  # <|fuzz|>
                    + lines[start_line - 1][start_col:end_col]
                    + f"<|/{tag}|>"  # </|fuzz|>
                    + lines[start_line - 1][end_col:]
                )
            else:
                lines[start_line - 1] = (
                    lines[start_line - 1][:start_col]
                    + f"<|{tag}|>"
                    + lines[start_line - 1][start_col:]
                )
                lines[end_line - 1] = (
                    lines[end_line - 1][:end_col]
                    + f"<|/{tag}|>"
                    + lines[end_line - 1][end_col:]
                )
        return "\n".join(lines)

    def train_test_split(
        self, val_split: float = 0.1, test_split: float = 0.15
    ) -> None:
        """
        Split the data into training, validation and test sets
        """
        assert self.processed_data is not None
        data = deepcopy(self.processed_data)
        np.random.shuffle(data)
        num_val = int(val_split * len(data))
        num_test = int(test_split * len(data))
        val_data = data[:num_val]
        test_data = data[num_val : num_val + num_test]
        train_data = data[num_val + num_test :]
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.logger.log("[green]Data is split![/green]")
        self.logger.log(
            f"Size of training data: {len(self.train_data)}, Validation data: {len(self.val_data)}, Test data: {len(self.test_data)}"
        )

    def truncate_code(self, src_code: str, branch: list) -> str:

        code_info = analyze_code(src_code)
        imports = ""
        for imp, pack in code_info["imports"]:
            packs = ", ".join(pack)
            if "from" in imp:
                imports += f"{imp} import {packs}\n"
            else:
                imports += f"{imp} {packs}\n"

        func_checked = []
        class_checked = {}

        for item in branch:
            for line in item:

                found = False
                # Get the functions containing the line
                for func in code_info["functions"]:
                    if (
                        (func["start_line"] <= line)
                        and (line <= func["end_line"])
                        and (func["name"] not in func_checked)
                    ):
                        # body += func["code"] + "\n\n"
                        func_checked.append(func["name"])
                        found = True
                if found:
                    continue

                # Get the class content that contains the line
                for class_item in code_info["classes"]:
                    if (class_item["start_line"] <= line) and (
                        line <= class_item["end_line"]
                    ):
                        # print(class_item['name'])
                        if class_item["name"] not in class_checked.keys():
                            class_check_info = {"class": True, "method_checked": []}
                        else:
                            class_check_info = class_checked[class_item["name"]]

                        for func in class_item["methods"]:
                            if (
                                (func["start_line"] <= line)
                                and (line <= func["end_line"])
                                and (
                                    func["name"]
                                    not in class_check_info["method_checked"]
                                )
                            ):
                                class_check_info["method_checked"].append(func["name"])
                                found = True
                        class_checked[class_item["name"]] = class_check_info
                if found:
                    continue

        body = ""
        for func in code_info["functions"]:
            if func["name"] in func_checked:
                body += func["code"] + "\n\n"

        for class_item in code_info["classes"]:

            if class_item["name"] in class_checked.keys():

                # print(class_item)
                body += class_item["code"] + "\n\n"
                # print(body)

                for func in class_item["methods"]:
                    if (
                        func["name"]
                        not in class_checked[class_item["name"]]["method_checked"]
                    ):
                        body = remove_method_from_class(
                            code=body,
                            class_name=class_item["name"],
                            method_name=func["name"],
                        )

        truncated_code = f"{imports}\n{body}"
        return truncated_code

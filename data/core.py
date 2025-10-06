import os
import ast
import dgl
import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from rich.progress import Progress
from rich.console import Console
from graph.core import Graph
from transformers import PreTrainedModel, PreTrainedTokenizer
from branch.utils import run_coverage, get_all_branch
from utils.utils import run_command, get_index_by_value, get_depth
from utils.code_analyzer import analyze_code, remove_method_from_class
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
from model.gnn import GRAPH_KEYS

# typing
from typing import List, Union, Dict, Any

PYNGUIN_TEMPLATE = """docker run --rm -v {}:/input:ro -v {}:/output -v {}:/package:ro {} \
    --module-name {} --coverage_metrics BRANCH --maximum_search_time {} --report-dir /output --project_path /input --output-path /output --output_variables TargetModule,CoverageTimeline --assertion-generation NONE"""

PROMPT_CODE = """Given a code script and an execution code lines, generate the test case for the corresponding code snippet:
```
{}
```

Here is the execution code lines:
{}
"""

PROMPT_CODE_TR = """Generate the test case for the code snippet:
```
{}
```
"""

PROMPT_GRAPH = """Generate the test case for the graph embedding of a targeted execution branch below:
{}
"""

PROMPT_CODE_GRAPH = """Generate the test case for the code below and the corresponding graph embedding of a targeted execution branch:

Here is the code:
```
{}
```

Here is the graph embedding:
{}
"""

RESPONSE_TEMPLATE = """Here is the test case:
```
{}
```
"""

PROMPT_COT = """Generate a test case for the following module such that:
- The test case use the pytest framework and executable.
- The test case will be put in the `tests/` directory which is place in the root of the project.
- The test case will need to execute the provided branch of execution in the provided module.

Here is the module:
```python
{module}
```

Here is the execution branch. The execution branch is a sequence of executable line number in the module:
{execution_branch}

THINK STEP-BY-STEP and provide your response in the following format:

```json
{{
  "test_case": <YOUR ANSWER FOR THE TEST CASE - JUST ONLY THE EXECUTABLE PYTHON CODE>
}}
```
"""

RESPONSE_BASELINE_TEMPLATE = """```json
{{
  "test_case": {}
}}
```
"""


class FuzzTagTransformer(ast.NodeTransformer):
    def __init__(self, tag: str):
        self.tag = tag

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, (int, float)):
                node.operand.value = (
                    f"<|{self.tag}|>-{node.operand.value}<|/{self.tag}|>"
                )
                return node.operand
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, str)):
            node.value = f"<|{self.tag}|>{node.value}<|/{self.tag}|>"
        return node


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
        model_name: str,
        feat_model: PreTrainedModel = None,
        feat_tokenizer: PreTrainedTokenizer = None,
        llm_tokenizer: PreTrainedTokenizer = None,
        coverage_image: str = "coverage",
        debug: bool = False,
        baseline_prompt: str = "code",
        graph_sampling: bool = False,
        max_tokens: int = 4096,
        n_hops: int = 2,
        gnn_mode: str = "node",
        data_fuzz: bool = False,
    ) -> None:
        self.name = name  # name of the data
        self.path = path  # path of the raw data
        self.logger = logger
        self.graph = graph
        self.num_cpu = num_cpu
        self.debug = debug
        self.model_name = model_name
        self.feat_model = feat_model
        self.feat_tokenizer = feat_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.coverage_image = coverage_image
        self.baseline_prompt = baseline_prompt
        self.graph_sampling = graph_sampling
        self.n_hops = n_hops
        self.max_tokens = max_tokens
        self.gnn_mode = gnn_mode
        self.data_fuzz = data_fuzz

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
        depth = get_depth(branch)
        if depth > 2:
            line_list = list(
                set(np.concatenate(np.array(branch, dtype=object)).tolist())
            )
        else:
            line_list = list(set(branch))
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

        processed_data = None
        processed_data_file_path = os.path.join(
            self.data_path,
            f"{self.baseline_prompt}_{self.max_tokens}_{self.model_name}",
            "processed_data.json",
        )
        if os.path.exists(processed_data_file_path):
            with open(
                processed_data_file_path,
                "r",
            ) as file:
                self.processed_data = json.load(file)
            processed_data = True
        else:
            processed_data_path = os.path.join(
                self.data_path,
                f"raw",
            )
            processed_prompt_path = os.path.join(
                self.data_path,
                f"{self.baseline_prompt}_{self.max_tokens}_{self.model_name}",
            )

            os.makedirs(processed_data_path, exist_ok=True)
            os.makedirs(processed_prompt_path, exist_ok=True)

        if processed_data:
            self.logger.log("[green]Data is already processed![/green]")
            self.logger.log(f"Size of data data: {len(self.processed_data)}")
            return

        with self.logger.status("[green]Preparing data...[/green]"):

            self.processed_data = {}
            num_tokens = []
            num_discarded = 0

            if ("graph" in self.baseline_prompt) and self.debug:
                graph_stats = {}
                for key in GRAPH_KEYS:
                    graph_stats[key] = {
                        "num_nodes": [],
                        "num_edges": [],
                        "in_max_degrees": [],
                        "out_max_degrees": [],
                        "in_min_degrees": [],
                        "out_min_degrees": [],
                        "num_components": [],
                    }

            for data_n in self.data.keys():

                self.processed_data[data_n] = {}

                for uuid, dat in self.data[data_n].items():
                    with open(dat["code_path"], "r") as file:
                        src_code = file.read()

                    mask = torch.load(dat["graph"]["mask_path"], weights_only=True)
                    self.logger.log(
                        f"Loaded mask for {uuid}: {len(mask)} and {len(dat['test_cases'])}"
                    )
                    assert len(mask) == len(dat["test_cases"])

                    if "graph" in self.baseline_prompt:
                        graph_name = f"{uuid}_graph.pt"
                        graph_path = os.path.join(processed_data_path, graph_name)

                        if os.path.exists(graph_path):
                            self.logger.log(
                                f"[yellow]Graph already exists for {uuid}, loading...[/yellow]"
                            )
                        else:
                            graph = self.read_graph(dat)

                            check_graph_exist_dict = {}
                            graph_dict = {}
                            for key in GRAPH_KEYS:
                                check_graph_exist_dict[key] = False

                            for key in graph.keys():
                                if isinstance(graph[key], dgl.DGLGraph):
                                    graph_dict[key] = graph[key]
                                    check_graph_exist_dict[key] = True

                            exist_atleast_one = False
                            for key in check_graph_exist_dict.keys():
                                if check_graph_exist_dict[key] == True:
                                    exist_atleast_one = True
                                    break

                            if not exist_atleast_one:
                                self.logger.log(
                                    f"[red]Graph is not generated for {uuid}[/red]"
                                )
                                num_discarded += len(dat["test_cases"])
                                continue
                            torch.save(graph_dict, graph_path)

                        if self.debug:
                            gstats = self.get_graph_stats(graph_dict)
                            for key in gstats.keys():
                                graph_stats[key]["num_nodes"].append(
                                    gstats[key]["num_nodes"]
                                )
                                graph_stats[key]["num_edges"].append(
                                    gstats[key]["num_edges"]
                                )
                                graph_stats[key]["in_max_degrees"].append(
                                    gstats[key]["in_max_degrees"]
                                )
                                graph_stats[key]["out_max_degrees"].append(
                                    gstats[key]["out_max_degrees"]
                                )
                                graph_stats[key]["in_min_degrees"].append(
                                    gstats[key]["in_min_degrees"]
                                )
                                graph_stats[key]["out_min_degrees"].append(
                                    gstats[key]["out_min_degrees"]
                                )
                                graph_stats[key]["num_components"].append(
                                    gstats[key]["num_components"]
                                )

                    for testcase in dat["test_cases"].keys():
                        test_code = dat["test_cases"][testcase]["test_case"]
                        if self.data_fuzz:
                            test_code = self.add_fuzz_tags(test_code)
                        if test_code == "N/A":
                            num_discarded += 1
                            continue
                        mask_key = int(testcase.split("_")[-1])
                        branch = mask[mask_key]
                        branch_line = dat["test_cases"][testcase]["branch"]
                        active_node = get_index_by_value(a=branch[0], val=1)
                        if active_node.size(0) == 0:
                            self.logger.log(
                                f"Active node empty at uuid: {uuid} testcase: {testcase}"
                            )
                            num_discarded += 1
                            continue

                        result = self.get_prompt(
                            src_code=src_code,
                            testcase_out=test_code,
                            mask=active_node,
                            tokenizer=self.llm_tokenizer,
                            branch=branch_line,
                            gnn_mode=self.gnn_mode,
                        )
                        if result is None:
                            num_discarded += 1
                            continue
                        prompt, response, full_text = result

                        num_token = len(self.llm_tokenizer.tokenize(full_text))
                        num_tokens.append(num_token)

                        if "graph" in self.baseline_prompt:
                            data = {
                                "uuid": f"{uuid}_{testcase}",
                                "prompt": prompt,
                                "response": response,
                                "full_text": full_text,
                                "active_node": active_node.tolist(),
                                "mask": mask[mask_key].tolist(),
                                "graph_path": graph_path,
                            }
                        else:
                            data = {
                                "uuid": f"{uuid}_{testcase}",
                                "prompt": prompt,
                                "response": response,
                                "full_text": full_text,
                                "active_node": None,
                                "mask": None,
                                "graph_path": None,
                            }

                        data_name = f"{uuid}_testcase_{testcase}.json"
                        data_path = os.path.join(processed_prompt_path, data_name)
                        with open(data_path, "w") as file:
                            json.dump(data, file, indent=4)

                        self.logger.log(
                            f"Data is saved to {data_path} for uuid - {uuid}, testcase - {testcase}"
                        )

                        self.processed_data[data_n][
                            f"{uuid}_testcase_{testcase}"
                        ] = data_path

        with open(processed_data_file_path, "w") as file:
            json.dump(self.processed_data, file, indent=4)

        self.logger.log("[green]Data is ready![/green]")
        self.logger.log(
            f"Size of processed data: {len(self.processed_data)}, num_discarded: {num_discarded}"
        )

        quartiles = np.quantile(num_tokens, [0, 0.25, 0.5, 0.75, 1])
        max_num_tokens = max(num_tokens)
        min_num_tokens = min(num_tokens)
        self.logger.log(
            f"Statistics of # tokens: {quartiles}, max: {max_num_tokens}, min: {min_num_tokens}, num_data: {len(num_tokens)}"
        )

        if "graph" in self.baseline_prompt and self.debug:
            for key in graph_stats.keys():
                self.logger.log(f"============= For graph {key}: =============")
                for skey in graph_stats[key].keys():
                    quartiles = np.quantile(
                        graph_stats[key][skey], [0, 0.25, 0.5, 0.75, 1]
                    )
                    max_num = max(graph_stats[key][skey])
                    min_num = min(graph_stats[key][skey])
                    self.logger.log(
                        f"Statistics of {skey}: {quartiles}, max: {max_num}, min: {min_num}, num_data: {len(graph_stats[key][skey])}"
                    )

    def prepare_data_for_test_gen(self):
        assert self.data is not None

        processed_data = None
        processed_data_file_path = os.path.join(
            self.data_path,
            f"{self.baseline_prompt}_{self.max_tokens}_{self.model_name}",
            "processed_data_for_test_gen.json",
        )
        if os.path.exists(processed_data_file_path):
            with open(
                processed_data_file_path,
                "r",
            ) as file:
                self.processed_data = json.load(file)
            processed_data = True
        else:
            processed_data_path = os.path.join(
                self.data_path,
                f"raw",
            )
            processed_prompt_path = os.path.join(
                self.data_path,
                f"{self.baseline_prompt}_{self.max_tokens}_{self.model_name}",
            )

            os.makedirs(processed_data_path, exist_ok=True)
            os.makedirs(processed_prompt_path, exist_ok=True)

        if processed_data:
            self.logger.log("[green]Data is already processed![/green]")
            self.logger.log(f"Size of data data: {len(self.processed_data)}")
            return

        with self.logger.status("[green]Preparing data for test gen...[/green]"):

            self.processed_data = {}
            num_tokens = []
            num_discarded = 0

            for data_n in self.data.keys():

                if "test" not in data_n:
                    continue

                self.processed_data[data_n] = {}

                for uuid, dat in self.data[data_n].items():
                    with open(dat["code_path"], "r") as file:
                        src_code = file.read()

                    branches = get_all_branch(code=src_code)

                    if "graph" in self.baseline_prompt:
                        graph_name = f"{uuid}_graph.pt"
                        graph_path = os.path.join(processed_data_path, graph_name)

                        if os.path.exists(graph_path):
                            self.logger.log(
                                f"[yellow]Graph already exists for {uuid}, loading...[/yellow]"
                            )
                            # load graph
                            with open(dat["graph"]["src_graph_path"], "r") as file:
                                graph = json.load(file)
                        else:
                            graph = self.read_graph(dat)
                            check_graph_exist_dict = {}
                            graph_dict = {}
                            for key in GRAPH_KEYS:
                                check_graph_exist_dict[key] = False

                            for key in graph.keys():
                                if isinstance(graph[key], dgl.DGLGraph):
                                    graph_dict[key] = graph[key]
                                    check_graph_exist_dict[key] = True

                            exist_atleast_one = False
                            for key in check_graph_exist_dict.keys():
                                if check_graph_exist_dict[key] == True:
                                    exist_atleast_one = True
                                    break

                            if not exist_atleast_one:
                                self.logger.log(
                                    f"[red]Graph is not generated for {uuid}[/red]"
                                )
                                num_discarded += len(dat["test_cases"])
                                continue
                            torch.save(graph_dict, graph_path)

                    for i, branch in enumerate(branches):
                        mask = self.get_mask_tensor(graph=graph, branch=branch)
                        assert len(mask.shape) == 2, f"Mask shape is {mask.shape}"
                        active_node = get_index_by_value(a=mask[0], val=1)
                        if active_node.size(0) == 0:
                            self.logger.log(
                                f"Active node empty at uuid: {uuid} for branch: {branch}"
                            )
                            num_discarded += 1
                            continue

                        result = self.get_prompt(
                            src_code=src_code,
                            testcase_out=None,
                            mask=active_node,
                            tokenizer=self.llm_tokenizer,
                            branch=branch,
                            gnn_mode=self.gnn_mode,
                            testing=True,
                        )
                        if result is None:
                            num_discarded += 1
                            continue
                        prompt = result

                        num_token = len(self.llm_tokenizer.tokenize(prompt))
                        num_tokens.append(num_token)

                        if "graph" in self.baseline_prompt:
                            data = {
                                "uuid": f"{uuid}_testcase_{i}",
                                "prompt": prompt,
                                "active_node": active_node.tolist(),
                                "mask": mask.tolist(),
                                "graph_path": graph_path,
                            }
                        else:
                            data = {
                                "uuid": f"{uuid}_testcase_{i}",
                                "prompt": prompt,
                                "active_node": None,
                                "mask": None,
                                "graph_path": None,
                            }

                        data_name = f"{uuid}_testcase_{i}.json"
                        data_path = os.path.join(processed_prompt_path, data_name)
                        with open(data_path, "w") as file:
                            json.dump(data, file, indent=4)

                        self.logger.log(
                            f"Data is saved to {data_path} for uuid - {uuid}, testcase - {i}"
                        )

                        self.processed_data[data_n][f"{uuid}_testcase_{i}"] = data_path

        with open(processed_data_file_path, "w") as file:
            json.dump(self.processed_data, file, indent=4)

        self.logger.log("[green]Data is ready![/green]")
        self.logger.log(
            f"Size of processed data: {len(self.processed_data)}, num_discarded: {num_discarded}"
        )

        quartiles = np.quantile(num_tokens, [0, 0.25, 0.5, 0.75, 1])
        max_num_tokens = max(num_tokens)
        min_num_tokens = min(num_tokens)
        self.logger.log(
            f"Statistics of # tokens: {quartiles}, max: {max_num_tokens}, min: {min_num_tokens}, num_data: {len(num_tokens)}"
        )

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
        gnn_mode: str = "graph",
        testing: bool = False,
    ):

        # self.logger.log(
        #     f"Preparing prompts with baseline_prompt: {self.baseline_prompt}"
        # )
        if not testing:
            if gnn_mode == "graph":
                graph_pad = "<|graph_pad|>"
            else:
                graph_pad = "<|graph_pad|>" * mask.size(0)
            if self.baseline_prompt == "code":
                code_line = self.generate_code_line(branch)
                text = PROMPT_CODE.format(src_code, code_line)
                response = RESPONSE_TEMPLATE.format(testcase_out)
            elif self.baseline_prompt == "graph":
                text = PROMPT_GRAPH.format(graph_pad)
                response = RESPONSE_TEMPLATE.format(testcase_out)
            elif self.baseline_prompt == "code_graph":
                text = PROMPT_CODE_GRAPH.format(src_code, graph_pad)
                response = RESPONSE_TEMPLATE.format(testcase_out)
            elif self.baseline_prompt == "code_tr":
                # self.logger.log("Truncating code...")
                trucated_code = self.truncate_code(src_code=src_code, branch=branch)
                if trucated_code is None:
                    self.logger.log("Truncated code is None")
                    return None
                text = PROMPT_CODE_TR.format(trucated_code)
                response = RESPONSE_TEMPLATE.format(testcase_out)
            elif self.baseline_prompt == "graph_tr":
                trucated_code = self.truncate_code(src_code=src_code, branch=branch)
                text = PROMPT_CODE_GRAPH.format(trucated_code, graph_pad)
                response = RESPONSE_TEMPLATE.format(testcase_out)
            elif self.baseline_prompt == "code_baseline":
                code_line = self.generate_code_line(branch)
                text = PROMPT_COT.format(module=src_code, execution_branch=code_line)
                response = RESPONSE_BASELINE_TEMPLATE.format(testcase_out)

            task_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
            )

            task_prompt_input = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
            )

            task_prompt_output = tokenizer.apply_chat_template(
                [{"role": "assistant", "content": response}],
                tokenize=False,
            )

            if len(self.llm_tokenizer.tokenize(task_prompt)) > self.max_tokens:
                self.logger.log(
                    f"[red]Task is too long: {len(self.llm_tokenizer.tokenize(task_prompt))} > {self.max_tokens}[/red]"
                )
                return None

            return task_prompt_input, task_prompt_output, task_prompt
        else:
            if gnn_mode == "graph":
                graph_pad = "<|graph_pad|>"
            else:
                graph_pad = "<|graph_pad|>" * mask.size(0)
            if self.baseline_prompt == "code":
                code_line = self.generate_code_line(branch)
                text = PROMPT_CODE.format(src_code, code_line)
            elif self.baseline_prompt == "graph":
                text = PROMPT_GRAPH.format(graph_pad)
            elif self.baseline_prompt == "code_graph":
                text = PROMPT_CODE_GRAPH.format(src_code, graph_pad)
            elif self.baseline_prompt == "code_tr":
                # self.logger.log("Truncating code...")
                trucated_code = self.truncate_code(src_code=src_code, branch=branch)
                if trucated_code is None:
                    self.logger.log("Truncated code is None")
                    return None
                text = PROMPT_CODE_TR.format(trucated_code)
            elif self.baseline_prompt == "graph_tr":
                trucated_code = self.truncate_code(src_code=src_code, branch=branch)
                text = PROMPT_CODE_GRAPH.format(trucated_code, graph_pad)
            elif self.baseline_prompt == "code_baseline":
                code_line = self.generate_code_line(branch)
                text = PROMPT_COT.format(module=src_code, execution_branch=code_line)

            task_prompt_input = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
            )

            if len(self.llm_tokenizer.tokenize(task_prompt_input)) > self.max_tokens:
                self.logger.log(
                    f"[red]Task is too long: {len(self.llm_tokenizer.tokenize(task_prompt_input))} > {self.max_tokens}[/red]"
                )
                return None

            return task_prompt_input

    def add_fuzz_tags(self, code: str, tag: str = "fuzz") -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return "N/A"

        transformer = FuzzTagTransformer(tag)
        tree = transformer.visit(tree)

        return ast.unparse(tree)

    def train_test_split(
        self, val_split: Union[float, int] = 0.1, test_only: bool = False
    ) -> None:
        """
        Split the data into training, validation and test sets
        """
        # TODO: Data splitting from the splitted directory must be implemented
        if test_only:
            self.logger.log("[green]Using only the test set, no need to split[/green]")
            test_data_by_project = self.processed_data["test_project"]
            test_data_by_module = self.processed_data["test_module"]

            self.test_data = {
                "project": test_data_by_project,
                "module": test_data_by_module,
            }
            self.logger.log("[green]Data is split![/green]")
            self.logger.log(f"Size of Test data: {len(self.test_data)}")
        else:
            self.logger.log(
                "[green]Splitting the data into training, validation and test sets...[/green]"
            )
            assert self.processed_data is not None
            # data = deepcopy(self.processed_data)
            # np.random.shuffle(data)
            num_val = (
                int(val_split * len(self.processed_data))
                if isinstance(val_split, float)
                else val_split
            )
            self.logger.log(f"Number of validation data: {num_val}")

            # split train and val
            keys_list = list(self.processed_data["train"].keys())
            np.random.shuffle(keys_list)
            val_keys = keys_list[:num_val]
            train_keys = keys_list[num_val:]

            train_data = {}
            val_data = {}
            test_data_by_project = self.processed_data["test_project"]
            test_data_by_module = self.processed_data["test_module"]
            for key in train_keys:
                train_data[key] = self.processed_data["train"][key]
            for key in val_keys:
                val_data[key] = self.processed_data["train"][key]

            self.train_data = train_data
            self.val_data = val_data
            self.test_data = {
                "project": test_data_by_project,
                "module": test_data_by_module,
            }
            self.logger.log("[green]Data is split![/green]")
            self.logger.log(
                f"Size of training data: {len(self.train_data)}, Validation data: {len(self.val_data)}, Test data: {len(self.test_data)}"
            )

    def truncate_code(self, src_code: str, branch: list) -> str:
        set_of_line = []
        depth = get_depth(branch)
        if depth > 1:
            for item in branch:
                set_of_line.extend(item)
            set_of_line = sorted(list(set(set_of_line)))
        else:
            set_of_line = sorted(list(set(branch)))

        code_line = src_code.split("\n")
        truncated_code = ""

        for i, line in enumerate(set_of_line):

            if code_line[line - 1].strip().startswith('"""'):
                continue
            if i == 0:
                truncated_code += code_line[line - 1]
                truncated_code += "\n"
            else:
                if line - 2 not in set_of_line:
                    indent = len(code_line[line - 1]) - len(
                        code_line[line - 1].lstrip()
                    )
                    truncated_code += " " * indent + "...\n"
                truncated_code += code_line[line - 1]
                truncated_code += "\n"
        return truncated_code

    def generate_code_line(self, branch):

        code_line = ""
        for item in branch:
            line = "->".join([str(i) for i in item])
            code_line += line + "\n"
        return code_line

    def get_graph_stats(self, graph_dict: Dict[str, dgl.DGLGraph]) -> dict:
        """
        Get the statistics of the graph
        """
        stats = {}
        for key in graph_dict.keys():
            graph = graph_dict[key]
            num_nodes = graph.num_nodes()
            num_edges = graph.num_edges()
            in_max_degrees = graph.in_degrees().float().max().item()
            out_max_degrees = graph.out_degrees().float().max().item()
            in_min_degrees = graph.in_degrees().float().max().item()
            out_min_degrees = graph.out_degrees().float().max().item()

            nx_graph = graph.to_networkx().to_undirected()
            num_components = nx.number_connected_components(nx_graph)

            stats[key] = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "in_max_degrees": in_max_degrees,
                "out_max_degrees": out_max_degrees,
                "in_min_degrees": in_min_degrees,
                "out_min_degrees": out_min_degrees,
                "num_components": num_components,
            }
        return stats

    def extract_blocks(self, source_code):
        """
        Parse source code and return:
            - blocks: list of dicts, each representing a code block
            - line_to_block: dict mapping each line number to the innermost block dict

        Each block dict is:
            {
                'type': 'function' or 'class',
                'name': str,
                'start': int,              # first line of the header/decorators
                'end': int,                # last line of block
                'header_lines': list[str], # header lines (decorators + def/class)
                'indent': int,             # indentation of header
                'parent': parent_block or None
            }
        """
        source_lines = source_code.splitlines()
        tree = ast.parse(source_code)
        blocks = []
        line_to_block = {}

        def visit(node, parent):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Collect decorators
                deco_lines = []
                for deco in getattr(node, "decorator_list", []):
                    for l in range(deco.lineno, node.lineno):
                        deco_lines.append(source_lines[l - 1])
                header_line = source_lines[node.lineno - 1]
                indent = len(header_line) - len(header_line.lstrip())
                # End line logic
                if hasattr(node, "end_lineno"):
                    block_end = node.end_lineno
                else:
                    block_end = node.body[-1].lineno if node.body else node.lineno
                block = {
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    "name": node.name,
                    "start": deco_lines[0] if deco_lines else node.lineno,
                    "end": block_end,
                    "header_lines": deco_lines + [header_line],
                    "indent": indent,
                    "parent": parent,
                }
                blocks.append(block)
                # Map all lines in this block to this block as innermost
                for i in range(node.lineno, block_end + 1):
                    line_to_block[i] = block
                # Recurse
                for child in ast.iter_child_nodes(node):
                    visit(child, block)

        for node in tree.body:
            visit(node, None)

        # Sort blocks by start line
        blocks.sort(key=lambda b: b["start"] if isinstance(b["start"], int) else 1)
        return blocks, line_to_block

    def extract_imports(self, source_code):
        """
        Parse source code and return all top-level import statements as a string
        (preserving order and original whitespace).
        """
        source_lines = source_code.splitlines()
        tree = ast.parse(source_code)
        import_lines = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Can handle multi-line imports (with parens/backslashes)
                start = node.lineno
                # Try to get end_lineno if available, fallback to start
                end = getattr(node, "end_lineno", start)
                import_lines.extend(source_lines[start - 1 : end])
        return "\n".join(import_lines)

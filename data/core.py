import os
import re
import ast
import json
import torch
import random
import anthropic
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import HeteroData
from rich.console import Console
from rich.pretty import pretty_repr
from graph.core import Graph
from transformers import PreTrainedModel, PreTrainedTokenizer
from branch.utils import run_coverage, get_all_branch
from utils.utils import run_command, get_index_by_value, get_depth
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy

# typing
from typing import List, Union, Dict, Optional

PROMPT_TEMPLATE = """# INSTRUCTION: You are an AI agent that generates executable Python test cases targeting a specific execution branch of a module.

Inputs:
- Module source: source code of the target module (Could be truncated to related line only).
- Execution branch information: the lines of the target module executed.
- Module path: a valid, importable path from the PYTHONPATH directory.
- Code Property Graph (CPG) embeddings (Optional): semantic and structural information about the code elements related to the branch.

Tasks:
1. Generate a runnable Python test file that executes the specified branch of the module.
2. Include meaningful assertions that confirm correct behavior and should pass for the given branch.
3. Output only the final, runnable Python test code—no explanations or reasoning text.

Requirements:
- All imports must be valid and correspond to existing modules; do not invent or hallucinate any packages.
- Use standard testing practices (unittest, pytest, or assert statements).
- Keep the code clear, minimal, and maintainable.

------------------------------------------------------------

# INPUTS:

## Module Source:
```
{}
```

## Execution Branches Information (Line to Line executed):
{}


## Module Path:
{}

## Code Property Graph (CPG) Node Embeddings:
{}

## Here's how to import the target module:
```
{}
```

------------------------------------------------------------"""

RESPONSE_TEMPLATE = """# OUTPUTS: 

Since the target module is imported from `{}`, the test code should be structured to correctly reference this module. Furthermore, the targeted module can be imported using the following code snippet:
```
{}
```

The test case should not hallucinate any imports and use valid functions/classes from the module.

Here is the generated Python test code targeting the specified execution branch
```
{}
```
"""

REASONING_TEMPLATE_PROMPT = """Given a data sample (including input and output), generate a reasoning explanation (within 100 words) that describes the analytical process for deriving the output from the input. This reasoning will be placed before the #OUTPUTS tag to train LLMs for reasoning.

Your reasoning should:
1. Explain how you analyze the input components (module source, execution branches, module path, CPG embeddings)
2. Describe how you synthesize this information to generate the output
3. Incorporate graph embedding information: each <|graph_pad|> represents a CPG node embedding for that branch. When embeddings are provided, explain how they encode semantic/structural relationships (e.g., control flow, data dependencies, variable interactions) that guide the output generation
4. Connect the execution branch patterns to the generated test code logic

Focus on the reasoning process, not describing what the test does.

# DATA SAMPLE:
{}

# OUTPUT INSTRUCTION: Return only the reasoning in this form:
````json
{{"reason": <YOUR ANSWER>}}
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
        llm_model_name: str,
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
        repo: str = None,
    ) -> None:
        self.name = name  # name of the data
        self.path = path  # path of the raw data
        self.logger = logger
        self.graph = graph
        self.num_cpu = num_cpu
        self.debug = debug
        self.model_name = model_name
        self.llm_model_name = llm_model_name
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
        self.repo = repo

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

    def get_mask_tensor(
        self, graph: Dict, branch: List[List[int]]
    ) -> List[torch.Tensor]:

        all_mask = []
        # branch_to_remove = []
        for j, branch_item in enumerate(branch):
            mask = np.zeros(len(graph["nodes"]))
            line_list = list(set(branch_item))
            for i in range(len(graph["nodes"])):
                node = graph["nodes"][i]
                # if node["location"]["filename"] == "N/A":
                try:
                    if node["properties"]["LINE_NUMBER"] in line_list:
                        mask[i] = 1
                except:
                    mask[i] = 0
            all_mask.append(mask)

        if len(all_mask) == 1:
            return None
        return all_mask

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

    def prepare_data_by_repo(self) -> None:
        assert self.processed_data is not None

        # take only data belong to a repo
        train_data = (
            self.processed_data["train"]
            if "train" in self.processed_data.keys()
            else None
        )
        test_modules = (
            self.processed_data["test_module"]
            if "test_module" in self.processed_data.keys()
            else None
        )
        test_project = (
            self.processed_data["test_project"]
            if "test_project" in self.processed_data.keys()
            else None
        )

        if train_data is not None:
            new_train_data = {}
            for key in train_data.keys():
                if self.repo in key:
                    new_train_data[key] = train_data[key]
        else:
            new_train_data = None

        if test_modules is not None:
            new_test_modules = {}
            for key in test_modules.keys():
                if self.repo in key:
                    new_test_modules[key] = test_modules[key]
        else:
            new_test_modules = None

        if test_project is not None:
            new_test_project = {}
            for key in test_project.keys():
                if self.repo in key:
                    new_test_project[key] = test_project[key]
        else:
            new_test_project = None

        self.processed_data = {
            "train": new_train_data if new_train_data is not None else {},
            "test_module": new_test_modules if new_test_modules is not None else {},
            "test_project": new_test_project if new_test_project is not None else {},
        }

    def prepare_data(self) -> None:
        """
        Prepare the training data for the model
        """
        assert self.data is not None

        processed_data = None
        if "graph" not in self.baseline_prompt:
            processed_data_file_path = os.path.join(
                self.data_path,
                f"{self.baseline_prompt}_{self.llm_model_name}",
                "original",
                "processed_data.json",
            )
        else:
            processed_data_file_path = os.path.join(
                self.data_path,
                f"{self.baseline_prompt}_{self.llm_model_name}_{self.gnn_mode}",
                "original",
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
                f"raw-pyg",
            )
            if "graph" not in self.baseline_prompt:
                processed_prompt_path = os.path.join(
                    self.data_path,
                    f"{self.baseline_prompt}_{self.llm_model_name}",
                    "original",
                )
            else:
                processed_prompt_path = os.path.join(
                    self.data_path,
                    f"{self.baseline_prompt}_{self.llm_model_name}_{self.gnn_mode}",
                    "original",
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

            for data_n in self.data.keys():  # train, test_module, test_project

                # load raw data
                raw_data_path = os.path.join(self.data_path, f"{data_n}.jsonl")

                if not os.path.exists(raw_data_path):
                    self.logger.log(
                        f"[red]Raw data path {raw_data_path} does not exist[/red]"
                    )
                    continue

                with open(raw_data_path, "r") as f:
                    raw_data = [json.loads(line) for line in f.readlines()]
                    raw_data_dict = {
                        item["id"]: item["local_imports"] for item in raw_data
                    }

                self.processed_data[data_n] = {}

                for uuid, dat in tqdm(
                    self.data[data_n].items(), position=0, leave=True
                ):

                    with open(dat["code_path"], "r") as file:
                        src_code = file.read()

                    local_imports = raw_data_dict[uuid]

                    module_path = dat.get("module_path", "N/A")
                    all_masks = torch.load(
                        dat["graph"]["mask_path"], weights_only=False
                    )
                    assert len(all_masks) == len(dat["test_cases"])

                    if "graph" in self.baseline_prompt:
                        graph_name = f"{uuid}_graph.pt"
                        graph_path = os.path.join(processed_data_path, graph_name)

                        if not os.path.exists(graph_path):
                            graph = self.read_graph(dat)
                            torch.save(graph, graph_path)
                        else:
                            graph = None

                    for testcase in dat["test_cases"].keys():
                        data_name = f"{uuid}_testcase_{testcase}.json"
                        data_path = os.path.join(processed_prompt_path, data_name)

                        with open(data_path, "r") as file:
                            data = json.load(file)
                        num_token = data["num_tokens"]
                        num_tokens.append(num_token)

                        self.processed_data[data_n][f"{uuid}_testcase_{testcase}"] = {
                            "num_tokens": num_token,
                            "path": data_path,
                        }

                        test_code = dat["test_cases"][testcase]["test_case"]
                        if self.data_fuzz:
                            test_code = self.add_fuzz_tags(test_code)
                        if test_code == "N/A":
                            num_discarded += 1
                            continue
                        mask_key = int(testcase.split("_")[-1])
                        branch_masks: List[torch.Tensor] = all_masks[mask_key]

                        if graph is not None:
                            for mask in branch_masks:
                                # print(mask.shape[0], graph.num_nodes)
                                assert (
                                    mask.shape[0] == graph.num_nodes
                                ), "Mask size mismatch!"
                        branch_line = dat["test_cases"][testcase]["branch"]
                        # print(branch_masks)
                        active_nodes = [
                            get_index_by_value(a=branch_masks[i], val=1)
                            for i in range(len(branch_masks))
                        ]
                        # print(active_nodes[0])
                        if len(active_nodes) == 0:
                            self.logger.log(
                                f"Active node empty at uuid: {uuid} testcase: {testcase}"
                            )
                            num_discarded += 1
                            continue

                        result = self.get_prompt(
                            src_code=src_code,
                            testcase_out=test_code,
                            active_nodes=active_nodes,
                            tokenizer=self.llm_tokenizer,
                            module_path=module_path,
                            branch=branch_line,
                            gnn_mode=self.gnn_mode,
                            local_imports=local_imports,
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
                                "active_node": [
                                    active_node.tolist() for active_node in active_nodes
                                ],
                                "mask": [
                                    all_masks[mask_key][i].tolist()
                                    for i in range(len(all_masks[mask_key]))
                                ],
                                "graph_path": graph_path,
                                "num_tokens": num_token,
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
                                "num_tokens": num_token,
                            }

                        with open(data_path, "w") as file:
                            json.dump(data, file, indent=4)

                        self.processed_data[data_n][f"{uuid}_testcase_{testcase}"] = {
                            "num_tokens": num_token,
                            "path": data_path,
                        }

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

    def prepare_reasoning_data(self) -> None:

        assert self.data is not None

        processed_data = False

        if "graph" not in self.baseline_prompt:
            processed_data_file_path = os.path.join(
                self.data_path,
                f"{self.baseline_prompt}_{self.llm_model_name}",
                "reasoning",
                "processed_data.json",
            )
        else:
            processed_data_file_path = os.path.join(
                self.data_path,
                f"{self.baseline_prompt}_{self.llm_model_name}_{self.gnn_mode}",
                "reasoning",
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
                f"raw-pyg",
            )
            if "graph" not in self.baseline_prompt:
                processed_prompt_path = os.path.join(
                    self.data_path,
                    f"{self.baseline_prompt}_{self.llm_model_name}",
                    "reasoning",
                )
            else:
                processed_prompt_path = os.path.join(
                    self.data_path,
                    f"{self.baseline_prompt}_{self.llm_model_name}_{self.gnn_mode}",
                    "reasoning",
                )

            os.makedirs(processed_data_path, exist_ok=True)
            os.makedirs(processed_prompt_path, exist_ok=True)

        if processed_data:
            self.logger.log("[green]Data is already processed![/green]")
            self.logger.log(f"Size of data data: {len(self.processed_data)}")
            return

        reasoning_path = os.path.join(
            self.data_path,
            "reasoning.jsonl",
        )
        assert os.path.exists(
            reasoning_path
        ), "Reasoning file not found, please generate it first."
        with open(reasoning_path, "r") as f:
            reasoning_data = [json.loads(line) for line in f.readlines()]

        reasoning_dict = {}
        for obj in reasoning_data:
            key = list(obj.keys())[0]
            reasoning_dict[key] = obj[key]

        generated_keys = sorted(
            list(set([list(obj.keys())[0] for obj in reasoning_data]))
        )

        with self.logger.status("[green]Preparing reasoning data...[/green]"):

            self.processed_data = {}
            num_tokens = []
            num_discarded = 0

            for data_n in self.data.keys():  # train, test_module, test_project

                self.processed_data[data_n] = {}

                for uuid, dat in tqdm(
                    self.data[data_n].items(), position=0, leave=True
                ):
                    with open(dat["code_path"], "r") as file:
                        src_code = file.read()

                    module_path = dat.get("module_path", "N/A")
                    all_masks = torch.load(
                        dat["graph"]["mask_path"], weights_only=False
                    )
                    assert len(all_masks) == len(dat["test_cases"])

                    if "graph" in self.baseline_prompt:
                        graph_name = f"{uuid}_graph.pt"
                        graph_path = os.path.join(processed_data_path, graph_name)

                        if not os.path.exists(graph_path):
                            graph = self.read_graph(dat)
                            torch.save(graph, graph_path)

                    for testcase in dat["test_cases"].keys():

                        if data_n == "train":

                            if f"{uuid}_testcase_{testcase}" not in generated_keys:
                                self.logger.log(
                                    f"Reasoning not found for uuid: {uuid} testcase: {testcase}"
                                )
                                num_discarded += 1
                                continue

                        test_code = dat["test_cases"][testcase]["test_case"]
                        if self.data_fuzz:
                            test_code = self.add_fuzz_tags(test_code)
                        if test_code == "N/A":
                            num_discarded += 1
                            continue
                        mask_key = int(testcase.split("_")[-1])
                        branch_masks: List[torch.Tensor] = all_masks[mask_key]
                        branch_line = dat["test_cases"][testcase]["branch"]
                        # print(branch_masks)
                        active_nodes = [
                            get_index_by_value(a=branch_masks[i], val=1)
                            for i in range(len(branch_masks))
                        ]
                        if len(active_nodes) == 0:
                            self.logger.log(
                                f"Active node empty at uuid: {uuid} testcase: {testcase}"
                            )
                            num_discarded += 1
                            continue

                        result = self.get_prompt(
                            src_code=src_code,
                            testcase_out=test_code,
                            active_nodes=active_nodes,
                            tokenizer=self.llm_tokenizer,
                            module_path=module_path,
                            branch=branch_line,
                            gnn_mode=self.gnn_mode,
                        )

                        if result is None:
                            num_discarded += 1
                            continue

                        prompt, response, full_text = result

                        if data_n == "train":
                            reasoning = reasoning_dict[f"{uuid}_testcase_{testcase}"]
                            if "<|graph_pad|>" in reasoning:
                                reasoning = reasoning.replace(
                                    "<|graph_pad|>", "CPG embedding(s)"
                                ).strip()

                            insert_text = f"\n\n<think>\n{reasoning}\n</think>\n"
                            text_before = full_text.split(
                                "Here is the generated Python test code targeting the specified execution branch"
                            )[0]
                            text_after = full_text.split(
                                "Here is the generated Python test code targeting the specified execution branch"
                            )[1]
                            new_full_text = (
                                text_before
                                + insert_text
                                + "From the above reason, here is the generated Python test code targeting the specified execution branch"
                                + text_after
                            )
                            full_text = new_full_text

                            text_before = response.split(
                                "Here is the generated Python test code targeting the specified execution branch"
                            )[0]
                            text_after = response.split(
                                "Here is the generated Python test code targeting the specified execution branch"
                            )[1]
                            new_response = (
                                text_before
                                + insert_text
                                + "From the above reason, here is the generated Python test code targeting the specified execution branch"
                                + text_after
                            )
                            response = new_response

                        num_token = len(self.llm_tokenizer.tokenize(full_text))
                        num_tokens.append(num_token)

                        if "graph" in self.baseline_prompt:
                            data = {
                                "uuid": f"{uuid}_{testcase}",
                                "prompt": prompt,
                                "response": response,
                                "full_text": full_text,
                                "active_node": [
                                    active_node.tolist() for active_node in active_nodes
                                ],
                                "mask": [
                                    all_masks[mask_key][i].tolist()
                                    for i in range(len(all_masks[mask_key]))
                                ],
                                "graph_path": graph_path,
                                "num_tokens": num_token,
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
                                "num_tokens": num_token,
                            }

                        data_name = f"{uuid}_testcase_{testcase}.json"
                        data_path = os.path.join(processed_prompt_path, data_name)
                        with open(data_path, "w") as file:
                            json.dump(data, file, indent=4)

                        self.processed_data[data_n][f"{uuid}_testcase_{testcase}"] = {
                            "num_tokens": num_token,
                            "path": data_path,
                        }

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

    def prepare_data_for_test_gen(self, branch_limit: int = 100) -> None:

        assert self.data is not None

        processed_data = None

        # load raw data
        raw_data_path = os.path.join(self.data_path, "test_module.jsonl")
        assert os.path.exists(raw_data_path), "Raw data for test generation not found!"

        with open(raw_data_path, "r") as f:
            raw_data = [json.loads(line) for line in f.readlines()]
            raw_data_dict = {item["id"]: item for item in raw_data}

        if "graph" not in self.baseline_prompt:
            processed_data_file_path = os.path.join(
                self.data_path,
                f"{self.baseline_prompt}_{self.llm_model_name}",
                "testgen",
                "processed_data_for_test_gen.json",
            )
        else:
            processed_data_file_path = os.path.join(
                self.data_path,
                f"{self.baseline_prompt}_{self.llm_model_name}_{self.gnn_mode}",
                "testgen",
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
                f"raw-pyg",
            )
            if "graph" not in self.baseline_prompt:
                processed_prompt_path = os.path.join(
                    self.data_path,
                    f"{self.baseline_prompt}_{self.llm_model_name}",
                    "testgen",
                )
            else:
                processed_prompt_path = os.path.join(
                    self.data_path,
                    f"{self.baseline_prompt}_{self.llm_model_name}_{self.gnn_mode}",
                    "testgen",
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
            num_branch_total = 0
            num_testcase_total = 0

            for data_n in self.data.keys():

                if "test" not in data_n:
                    continue

                self.processed_data[data_n] = {}

                for uuid, dat in self.data[data_n].items():

                    with open(dat["code_path"], "r") as file:
                        src_code = file.read()

                    local_imports = raw_data_dict[uuid]["local_imports"]

                    module_path = dat.get("module_path", "N/A")
                    branches = get_all_branch(code=src_code, branch_limit=branch_limit)
                    num_branch_total += len(branches)

                    if "graph" in self.baseline_prompt:
                        graph_name = f"{uuid}_graph.pt"
                        graph_path = os.path.join(processed_data_path, graph_name)

                        if os.path.exists(graph_path):
                            with open(dat["graph"]["src_graph_path"], "r") as file:
                                graph = json.load(file)
                        else:
                            graph = self.read_graph(dat)
                            torch.save(graph, graph_path)

                    for i, branch in enumerate(branches):

                        all_masks = self.get_mask_tensor(graph=graph, branch=branch)
                        assert len(branch) == len(
                            all_masks
                        ), "Mask and branch length mismatch: {} vs {}".format(
                            len(all_masks), len(branch)
                        )

                        if all_masks is None:
                            self.logger.log(
                                f"Only import branch at uuid: {uuid}, testcase: {i}"
                            )

                        active_nodes = [
                            get_index_by_value(a=all_masks[j], val=1)
                            for j in range(len(all_masks))
                        ]

                        result = self.get_prompt(
                            src_code=src_code,
                            testcase_out=None,
                            active_nodes=active_nodes,
                            tokenizer=self.llm_tokenizer,
                            module_path=module_path,
                            branch=branch,
                            gnn_mode=self.gnn_mode,
                            testing=True,
                            local_imports=local_imports,
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
                                "active_node": [
                                    active_node.tolist() for active_node in active_nodes
                                ],
                                "mask": [
                                    all_masks[i].tolist() for i in range(len(all_masks))
                                ],
                                "graph_path": graph_path,
                                "num_tokens": num_token,
                                "branch": branch,
                                "module_path": module_path,
                                "code_path": dat["code_path"],
                            }

                        else:
                            data = {
                                "uuid": f"{uuid}_testcase_{i}",
                                "prompt": prompt,
                                "active_node": None,
                                "mask": None,
                                "graph_path": None,
                                "num_tokens": num_token,
                                "branch": None,
                                "module_path": module_path,
                            }

                        data_name = f"{uuid}_testcase_{i}.json"
                        data_path = os.path.join(processed_prompt_path, data_name)
                        with open(data_path, "w") as file:
                            json.dump(data, file, indent=4)
                            print(f"Saved data to {data_path}")

                        self.processed_data[data_n][f"{uuid}_testcase_{i}"] = {
                            "num_tokens": num_token,
                            "path": data_path,
                        }
                        num_testcase_total += 1

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

    def filter_by_max_min_tokens(self, max_tokens: int, min_tokens: int) -> None:
        self.logger.log(f"[green]Filtering data by max tokens {max_tokens}...[/green]")
        assert self.processed_data is not None
        filtered_data = {}
        need_to_save = False
        for data_n in self.processed_data.keys():
            filtered_data[data_n] = {}
            for key in tqdm(self.processed_data[data_n].keys()):
                if isinstance(self.processed_data[data_n][key], dict):
                    if (
                        self.processed_data[data_n][key]["num_tokens"] <= max_tokens
                    ) and (
                        self.processed_data[data_n][key]["num_tokens"] >= min_tokens
                    ):
                        filtered_data[data_n][key] = self.processed_data[data_n][key]
                else:
                    path = self.processed_data[data_n][key]
                    with open(path, "r") as file:
                        data = json.load(file)
                        num_token = data["num_tokens"]
                    if (num_token <= max_tokens) and (num_token >= min_tokens):
                        filtered_data[data_n][key] = self.processed_data[data_n][key]
                    self.processed_data[data_n][key] = {
                        "num_tokens": num_token,
                        "path": path,
                    }
                    need_to_save = True

        if need_to_save:
            processed_data_file_path = os.path.join(
                self.data_path,
                f"{self.baseline_prompt}_{self.llm_model_name}_{self.gnn_mode}",
                "processed_data.json",
            )
            with open(processed_data_file_path, "w") as file:
                json.dump(self.processed_data, file, indent=4)

        self.processed_data = filtered_data
        self.logger.log(
            f"[green]Data is filtered by max tokens {max_tokens}! New size: train - {len(self.processed_data['train'])} - test {len(self.processed_data['test_module'])}[/green]"
        )

    def sample_for_reasoning(self, max_samples: int) -> None:
        self.logger.log(f"[green]Filtering data for reasoning...[/green]")
        assert self.processed_data is not None
        filtered_data = {}
        data_n = "train"
        filtered_data = {}

        # get num data point by repo:
        total_data = len(self.processed_data[data_n].keys())
        repo_stats = {}
        data_point_by_repo = {}
        for key in self.processed_data[data_n].keys():
            repo = key.split("-")[0]
            if repo not in repo_stats.keys():
                repo_stats[repo] = 0
            if repo not in data_point_by_repo.keys():
                data_point_by_repo[repo] = []
            repo_stats[repo] += 1
            data_point_by_repo[repo].append(key)

        self.logger.log(f"Total data points: {total_data}")
        self.logger.log(f"Data points by repo: {pretty_repr(repo_stats)}")

        for repo in data_point_by_repo.keys():
            repo_data_points = data_point_by_repo[repo]
            num_data_points = len(repo_data_points)
            num_samples = int((num_data_points / total_data) * max_samples)
            self.logger.log(
                f"Randomly selecting {num_samples} samples from repo {repo} with {num_data_points} data points"
            )
            selected_data_points = random.sample(
                repo_data_points, min(num_samples, num_data_points)
            )
            for data_point in selected_data_points:
                filtered_data[data_point] = self.processed_data[data_n][data_point]

        self.processed_data[data_n] = filtered_data
        self.logger.log(
            f"[green]Data is filtered for reasoning! New size: {len(filtered_data)}[/green]"
        )

    def add_reasoning(self, reasoning_dict: dict) -> None:
        # add reasoning to the training data
        self.logger.log(f"[green]Adding reasoning to the data...[/green]")
        assert self.processed_data is not None

        samples = self.processed_data["train"]
        for key in tqdm(samples.keys()):
            if key in reasoning_dict.keys():
                data_path = samples[key]["path"]
                with open(data_path, "r") as file:
                    data = json.load(file)
                reasoning = reasoning_dict[key]

                # processing full text
                full_text = data["full_text"]
                insert_text = f"\n\n# Thinking:\n<think>\n{reasoning}\n</think>\n"
                text_before = full_text.split(
                    "Here is the generated Python test code targeting the specified execution branch"
                )[0]
                text_after = full_text.split(
                    "Here is the generated Python test code targeting the specified execution branch"
                )[1]
                new_full_text = (
                    text_before
                    + insert_text
                    + "From the above reason, here is the generated Python test code targeting the specified execution branch"
                    + text_after
                )
                data["full_text"] = new_full_text
                # save data
                with open(data_path, "w") as file:
                    json.dump(data, file, indent=4)

    def read_graph(self, data: dict) -> dict:

        graph_path = data["graph"]["src_graph_path"]
        with open(graph_path, "r") as file:
            graph = json.load(file)

        # graph_dict = {}
        num_nodes = len(graph["nodes"])
        feat = torch.load(data["graph"]["node_feature_path"], weights_only=True)

        assert num_nodes == feat.shape[0]

        graph_data = HeteroData()
        graph_data["node"].x = feat
        edge_dict = self.read_edge(graph)

        for etype in edge_dict.keys():
            u = torch.Tensor(edge_dict[etype][0]).long()  # Size (num_edges,)
            v = torch.Tensor(edge_dict[etype][1]).long()  # Size (num_edges,)
            graph_data["node", etype, "node"].edge_index = torch.stack(
                [u, v], dim=0
            )  # Size (2, num_edges)
            # graph = dgl.graph((u, v), num_nodes=num_nodes)
            # graph.ndata["feat"] = feat
            # graph_dict[etype] = graph
        # graph_dict["num_nodes"] = num_nodes
        # graph_dict["feat_size"] = feat.size()
        # return graph_dict
        # print(
        #     f"Number of nodes in graph: {graph_data.num_nodes}, num nodes: {num_nodes}"
        # )
        assert graph_data.num_nodes == num_nodes, "Number of nodes mismatch!"
        return graph_data

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
        active_nodes: List[torch.Tensor],
        branch: List,
        module_path: str,
        tokenizer: PreTrainedTokenizer,
        gnn_mode: str = "branch",
        testing: bool = False,
        local_imports: Optional[List[str]] = None,
    ):

        # self.logger.log(
        #     f"Preparing prompts with baseline_prompt: {self.baseline_prompt}"
        # )
        if not testing:

            # Extract imports from testcase_out
            if local_imports is not None:
                import_lines = "\n".join(local_imports)
            else:
                try:
                    tree = ast.parse(testcase_out)
                    import_lines = []

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            # Handle: import module, import module as alias
                            import_lines.append(ast.unparse(node))
                        elif isinstance(node, ast.ImportFrom):
                            # Handle: from module import name, from module import name as alias
                            import_lines.append(ast.unparse(node))

                    import_lines = "\n".join(import_lines)
                except Exception as e:
                    # Fallback: regex-based extraction if AST parsing fails
                    import re

                    import_pattern = r"^(?:from\s+[\w.]+\s+)?import\s+.+$"
                    lines = testcase_out.split("\n")
                    import_lines = [
                        line for line in lines if re.match(import_pattern, line.strip())
                    ]
                    import_lines = "\n".join(import_lines)

            if gnn_mode == "branch":
                graph_pad = ""
                for i, item in enumerate(active_nodes):
                    if i == 0:
                        if len(active_nodes) >= 1:
                            graph_pad += "Import branch: <|graph_pad|>" + "\n"
                        else:
                            graph_pad += "Import branch: Not Available" + "\n"
                    else:
                        if len(active_nodes) >= 1:
                            graph_pad += f"Branch #{i}: <|graph_pad|>\n"
                        else:
                            graph_pad += f"Branch #{i}: Not Available\n"
            else:
                graph_pad = ""
                for i, item in enumerate(active_nodes):
                    if i == 0:
                        if len(active_nodes) >= 1:
                            graph_pad += (
                                "Import branch: "
                                + "<|graph_pad|>" * item.size(0)
                                + "\n"
                            )
                        else:
                            graph_pad += "Import branch: Not Available" + "\n"
                    else:
                        if len(active_nodes) >= 1:
                            graph_pad += (
                                f"Branch #{i}: " + "<|graph_pad|>" * item.size(0) + "\n"
                            )
                        else:
                            graph_pad += f"Branch #{i}: Not Available\n"
            branch_line = ""
            for i, branch_item in enumerate(branch):
                if i == 0:
                    branch_line += (
                        f"Import branch: "
                        + "->".join([str(item) for item in branch_item])
                        + "\n"
                    )
                    continue
                branch_line += (
                    f"Branch #{i}: "
                    + "->".join([str(item) for item in branch_item])
                    + "\n"
                )
            if self.baseline_prompt == "code":
                text = PROMPT_TEMPLATE.format(
                    src_code, branch_line, module_path, "Not Available", import_lines
                )
                response = RESPONSE_TEMPLATE.format(
                    module_path, import_lines, testcase_out
                )
            elif self.baseline_prompt == "graph":
                text = PROMPT_TEMPLATE.format(
                    "Not Available",
                    "Not Available",
                    module_path,
                    graph_pad,
                    import_lines,
                )
                response = RESPONSE_TEMPLATE.format(
                    module_path, import_lines, testcase_out
                )
            elif self.baseline_prompt == "code_graph":
                text = PROMPT_TEMPLATE.format(
                    src_code, branch_line, module_path, graph_pad, import_lines
                )
                response = RESPONSE_TEMPLATE.format(
                    module_path, import_lines, testcase_out
                )
            elif self.baseline_prompt == "code_tr":
                truncated_code = self.truncate_code(src_code=src_code, branch=branch)
                if truncated_code is None:
                    self.logger.log("Truncated code is None")
                    return None
                text = PROMPT_TEMPLATE.format(
                    truncated_code,
                    branch_line,
                    module_path,
                    "Not Available",
                    import_lines,
                )
                response = RESPONSE_TEMPLATE.format(
                    module_path, import_lines, testcase_out
                )
            elif self.baseline_prompt == "graph_tr":
                truncated_code = self.truncate_code(src_code=src_code, branch=branch)
                text = PROMPT_TEMPLATE.format(
                    truncated_code, branch_line, module_path, graph_pad, import_lines
                )
                response = RESPONSE_TEMPLATE.format(
                    module_path, import_lines, testcase_out
                )

            task_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
            )

            task_prompt_input = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": text},
                ],
                tokenize=False,
            )

            task_prompt_output = tokenizer.apply_chat_template(
                [{"role": "assistant", "content": response}],
                tokenize=False,
            )

            # if len(self.llm_tokenizer.tokenize(task_prompt)) > self.max_tokens:
            #     self.logger.log(
            #         f"[red]Task is too long: {len(self.llm_tokenizer.tokenize(task_prompt))} > {self.max_tokens}[/red]"
            #     )
            #     return None

            return task_prompt_input, task_prompt_output, task_prompt
        else:
            # Extract imports from testcase_out

            if local_imports is not None:
                import_lines = "\n".join(local_imports)
            else:
                try:
                    tree = ast.parse(testcase_out)
                    import_lines = []

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            # Handle: import module, import module as alias
                            import_lines.append(ast.unparse(node))
                        elif isinstance(node, ast.ImportFrom):
                            # Handle: from module import name, from module import name as alias
                            import_lines.append(ast.unparse(node))

                    import_lines = "\n".join(import_lines)
                except (SyntaxError, ValueError):
                    # Fallback: regex-based extraction if AST parsing fails
                    import re

                    import_pattern = r"^(?:from\s+[\w.]+\s+)?import\s+.+$"
                    lines = testcase_out.split("\n")
                    import_lines = [
                        line for line in lines if re.match(import_pattern, line.strip())
                    ]
                    import_lines = "\n".join(import_lines)

            if gnn_mode == "branch":
                graph_pad = ""
                for i, item in enumerate(active_nodes):
                    if i == 0:
                        if len(active_nodes) >= 1:
                            graph_pad += "Import branch: <|graph_pad|>" + "\n"
                        else:
                            graph_pad += "Import branch: Not Available" + "\n"
                    else:
                        if len(active_nodes) >= 1:
                            graph_pad += f"Branch #{i}: <|graph_pad|>\n"
                        else:
                            graph_pad += f"Branch #{i}: Not Available\n"
            else:
                graph_pad = ""
                for i, item in enumerate(active_nodes):
                    if i == 0:
                        if len(active_nodes) >= 1:
                            graph_pad += (
                                "Import branch: "
                                + "<|graph_pad|>" * item.size(0)
                                + "\n"
                            )
                        else:
                            graph_pad += "Import branch: Not Available" + "\n"
                    else:
                        if len(active_nodes) >= 1:
                            graph_pad += (
                                f"Branch #{i}: " + "<|graph_pad|>" * item.size(0) + "\n"
                            )
                        else:
                            graph_pad += f"Branch #{i}: Not Available\n"
            branch_line = ""
            for i, branch_item in enumerate(branch):
                if i == 0:
                    branch_line += (
                        f"Import branch: "
                        + "->".join([str(item) for item in branch_item])
                        + "\n"
                    )
                    continue
                branch_line += (
                    f"Branch #{i}: "
                    + "->".join([str(item) for item in branch_item])
                    + "\n"
                )
            if self.baseline_prompt == "code":
                text = PROMPT_TEMPLATE.format(
                    src_code, branch_line, module_path, "Not Available", import_lines
                )
            elif self.baseline_prompt == "graph":
                text = PROMPT_TEMPLATE.format(
                    "Not Available",
                    "Not Available",
                    module_path,
                    graph_pad,
                    import_lines,
                )
            elif self.baseline_prompt == "code_graph":
                text = PROMPT_TEMPLATE.format(
                    src_code, branch_line, module_path, graph_pad, import_lines
                )
            elif self.baseline_prompt == "code_tr":
                truncated_code = self.truncate_code(src_code=src_code, branch=branch)
                if truncated_code is None:
                    self.logger.log("Truncated code is None")
                    return None
                text = PROMPT_TEMPLATE.format(
                    truncated_code,
                    branch_line,
                    module_path,
                    "Not Available",
                    import_lines,
                )
            elif self.baseline_prompt == "graph_tr":
                truncated_code = self.truncate_code(src_code=src_code, branch=branch)
                text = PROMPT_TEMPLATE.format(
                    truncated_code, branch_line, module_path, graph_pad, import_lines
                )

            task_prompt_input = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": text},
                ],
                tokenize=False,
            )

            # if len(self.llm_tokenizer.tokenize(task_prompt_input)) > self.max_tokens:
            #     self.logger.log(
            #         f"[red]Task is too long: {len(self.llm_tokenizer.tokenize(task_prompt_input))} > {self.max_tokens}[/red]"
            #     )
            #     return None

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
            test_data_by_project = (
                self.processed_data["test_project"]
                if "test_project" in self.processed_data.keys()
                else {}
            )
            test_data_by_module = (
                self.processed_data["test_module"]
                if "test_module" in self.processed_data.keys()
                else {}
            )

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
            num_val = val_split
            self.logger.log(f"Number of validation data: {num_val}")

            # split train and val
            keys_list = list(self.processed_data["train"].keys())
            repo_dict = {}
            for key in keys_list:
                repo = key.split("-")[0]
                if repo not in repo_dict:
                    repo_dict[repo] = []
                repo_dict[repo].append(key)

            num_repo = len(repo_dict.keys())
            num_val_per_repo = max(1, num_val // num_repo)

            val_keys = []
            for repo in repo_dict.keys():
                repo_keys = repo_dict[repo]
                np.random.shuffle(repo_keys)
                val_keys.extend(repo_keys[:num_val_per_repo])

            train_keys = [key for key in keys_list if key not in val_keys]

            train_data = {}
            val_data = {}
            test_data_by_project = (
                self.processed_data["test_project"]
                if "test_project" in self.processed_data.keys()
                else {}
            )
            test_data_by_module = (
                self.processed_data["test_module"]
                if "test_module" in self.processed_data.keys()
                else {}
            )
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

    # def get_graph_stats(self, graph_dict: Dict[str, dgl.DGLGraph]) -> dict:
    #     """
    #     Get the statistics of the graph
    #     """
    #     stats = {}
    #     for key in graph_dict.keys():
    #         graph = graph_dict[key]
    #         num_nodes = graph.num_nodes()
    #         num_edges = graph.num_edges()
    #         in_max_degrees = graph.in_degrees().float().max().item()
    #         out_max_degrees = graph.out_degrees().float().max().item()
    #         in_min_degrees = graph.in_degrees().float().max().item()
    #         out_min_degrees = graph.out_degrees().float().max().item()

    #         nx_graph = graph.to_networkx().to_undirected()
    #         num_components = nx.number_connected_components(nx_graph)

    #         stats[key] = {
    #             "num_nodes": num_nodes,
    #             "num_edges": num_edges,
    #             "in_max_degrees": in_max_degrees,
    #             "out_max_degrees": out_max_degrees,
    #             "in_min_degrees": in_min_degrees,
    #             "out_min_degrees": out_min_degrees,
    #             "num_components": num_components,
    #         }
    #     return stats

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


def get_reasoning(
    samples: Dict[str, dict],
    api_key: str,
    model: str,
    max_tokens: int = 512,
    console: Console = None,
    temperature: float = 0.7,
    save_path: str = None,
) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    reason_dict = {}
    with open(save_path, "a") as file:
        for key in tqdm(samples.keys()):
            data_path = samples[key]["path"]
            with open(data_path, "r") as f:
                sample = json.load(f)
            full_text = sample["full_text"]
            prompt = REASONING_TEMPLATE_PROMPT.format(full_text)
            # Anthropic's messages.create API
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,  # Anthropic uses max_tokens, not max_output_tokens
                "temperature": temperature,
                "messages": messages,
            }
            response = client.messages.create(**kwargs)

            fence = re.compile(
                r"```(?:json)?\s*([\s\S]*?\{[\s\S]*?\})\s*```", re.MULTILINE
            )
            m = fence.search(response.content[0].text)
            payload = m.group(1) if m else response.content[0].text

            # 3) Try JSON first:
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = payload
                print(f"Failed to extract json payload: {payload}")

            if ("reason" not in data) or (not isinstance(data, dict)):
                print(f"Failed to extract reason in payload: {payload}")
                reason_dict[key] = data
            else:
                reason_dict[key] = data["reason"]
            console.log(
                f"[green]Reasoning generated for sample {key}[/green]: {reason_dict[key]}"
            )
            file.write(json.dumps({key: reason_dict[key]}))
            file.write("\n")
            console.log(f"[yellow]Reasoning saved to {save_path}[/yellow]")
    return reason_dict

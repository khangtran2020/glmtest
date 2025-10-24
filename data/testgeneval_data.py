import os
import ast
import json
import time
import torch
import shutil
from typing import List
from rich.console import Console
from rich.progress import Progress
from data.core import Data
from graph.core import Graph
from transformers import PreTrainedModel, PreTrainedTokenizer

KEY_ID = "id"
NEW_KEY_ID = "uuid"


class TestGenEval(Data):

    def __init__(
        self,
        logger: Console,
        path: str,
        graph: Graph,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        llm_tokenizer: PreTrainedTokenizer,
        model_name: str,
        debug: bool = False,
        baseline_prompt: str = "code",
        graph_sampling: bool = False,
        max_tokens: int = 512,
        gnn_mode: str = "node",
        raw_overwrite: bool = False,
        repo: str = None,
        **kwargs,
    ) -> None:
        self.name = "TestGenEval"
        super().__init__(
            name=self.name,
            path=path,
            logger=logger,
            graph=graph,
            feat_model=model,
            feat_tokenizer=tokenizer,
            llm_tokenizer=llm_tokenizer,
            num_cpu=-1,
            debug=debug,
            model_name=model_name,
            baseline_prompt=baseline_prompt,
            graph_sampling=graph_sampling,
            max_tokens=max_tokens,
            gnn_mode=gnn_mode,
            repo=repo,
            **kwargs,
        )
        self.data_path = os.path.join(path, self.name)
        self.debug = debug
        self.data = None
        self.repo = repo
        self.raw_overwrite = raw_overwrite
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            self.logger.log(
                f"Data path not found, created a new one at {self.data_path}"
            )
        else:
            self.logger.log(f"Data path found at {self.data_path}")
            if os.path.exists(os.path.join(self.data_path, "data_processed.json")):
                with open(
                    os.path.join(self.data_path, "data_processed.json"), "r"
                ) as file:
                    self.data = json.load(file)
            else:
                if not os.path.exists(os.path.join(self.data_path, "data.jsonl")):
                    logger.log("data.jsonl not found, please crawl the data")
                else:
                    logger.log(
                        "Found data.jsonl file, but not processed. PLEASE RUN `process_raw`"
                    )
        if self.feat_model is not None:
            self.logger.log(
                f"Initialized {self.name} dataset, with model on device: {self.feat_model.device}"
            )

    def crawl(self) -> None:
        info = f"""To crawl this data, please do as following:
1. PLEASE RUN THE EXPERIMENT MANUALLY AS IN THE `testgeneval_pipline`
2. After that, please copy the final `.jsonl` file to the `{self.data_path}` folder
"""
        self.logger.log(info)

    def process_raw(self) -> None:

        process = False
        # check if the data has been processeds
        if os.path.exists(os.path.join(self.data_path, "data_processed.json")):
            self.logger.log(
                "Found data_processed.json file, do not need to process raw data"
            )
            # load data
            with open(os.path.join(self.data_path, "data_processed.json"), "r") as file:
                self.data = json.load(file)
        else:
            process = True

        if not process:
            return

        if not os.path.exists(os.path.join(self.data_path, "train.jsonl")):
            raise FileNotFoundError("train.jsonl not found, please crawl the data")

        if not os.path.exists(os.path.join(self.data_path, "test_project.jsonl")):
            self.logger.log("test_project.jsonl not found, please crawl the data")

        if not os.path.exists(os.path.join(self.data_path, "test_module.jsonl")):
            raise FileNotFoundError(
                "test_module.jsonl not found, please crawl the data"
            )

        code_path = os.path.join(self.data_path, "codes")
        graph_path = os.path.join(self.data_path, "graphs")

        # make projects dir
        if self.raw_overwrite:

            if os.path.exists(code_path):
                shutil.rmtree(code_path)
            if os.path.exists(graph_path):
                shutil.rmtree(graph_path)

            os.makedirs(code_path)
            os.makedirs(graph_path)

        data_dict = {}

        repos = []
        num_module = 0

        data_name = ["train", "test_project", "test_module"]
        for data_n in data_name:

            if not os.path.exists(os.path.join(self.data_path, f"{data_n}.jsonl")):
                continue

            with open(os.path.join(self.data_path, f"{data_n}.jsonl"), "r") as file:
                raw_data = [json.loads(l) for l in file.readlines()]

            for i, task in enumerate(raw_data):
                if NEW_KEY_ID not in task.keys():
                    task[NEW_KEY_ID] = task["instance_id"]

            raw_data = {task[NEW_KEY_ID]: task for task in raw_data}

            with Progress() as progress:
                task = progress.add_task("[cyan]Processing...", total=len(raw_data))

                data = []

                for i, key in enumerate(raw_data.keys()):
                    start_time = time.time()
                    dat = {}
                    dat["test_cases"] = {}
                    dat["graph"] = {}
                    dat["uuid"] = key
                    dat["code_path"] = os.path.join(
                        code_path, f"{raw_data[key][NEW_KEY_ID]}.py"
                    )
                    dat["graph"]["src_graph_path"] = os.path.join(
                        graph_path, f"{raw_data[key][NEW_KEY_ID]}.json"
                    )
                    dat["graph"]["node_feature_path"] = os.path.join(
                        graph_path, f"{raw_data[key][NEW_KEY_ID]}.pt"
                    )
                    dat["graph"]["mask_path"] = os.path.join(
                        graph_path, f"{raw_data[key][NEW_KEY_ID]}_mask.pt"
                    )
                    # dat["repo"] = raw_data[key]["repo"]
                    dat["module_path"] = raw_data[key]["code_file"]

                    if (not os.path.exists(dat["code_path"])) or (
                        os.path.exists(dat["code_path"]) and self.raw_overwrite
                    ):
                        with open(dat["code_path"], "w") as file:
                            file.write(raw_data[key]["code_src"])

                    if (
                        os.path.exists(dat["graph"]["mask_path"])
                        and os.path.exists(dat["graph"]["node_feature_path"])
                        and (not self.raw_overwrite)
                    ):
                        mask = torch.load(dat["graph"]["mask_path"])
                        node_feat = torch.load(dat["graph"]["node_feature_path"])

                        # check if size is zero
                        if len(mask) == 0:
                            self.logger.log(f"[red]Mask is empty for {key}[/red]")

                        if node_feat.size(0) == 0:
                            self.logger.log(
                                f"[red]Node features are empty for {key}[/red]"
                            )

                        if (len(mask) != 0) and (node_feat.size(0) != 0):
                            idx = 0
                            for i, tkey in enumerate(
                                raw_data[key]["test_cases"].keys()
                            ):
                                if raw_data[key]["branches"][tkey] == []:
                                    continue
                                if raw_data[key]["test_cases"][tkey] == "":
                                    continue
                                try:
                                    ast.parse(raw_data[key]["test_cases"][tkey]["code"])
                                except Exception as e:
                                    continue
                                nkey = f"test_case_{idx}"
                                dat["test_cases"][nkey] = {}
                                dat["test_cases"][nkey]["test_case"] = raw_data[key][
                                    "test_cases"
                                ][tkey]["code"]
                                dat["test_cases"][nkey]["branch"] = raw_data[key][
                                    "branches"
                                ][tkey]
                                idx += 1
                            data.append(dat)
                            repos.append(raw_data[key]["repo"])
                            num_module += 1
                            end_time = time.time()
                            self.logger.log(
                                f"Processed module {num_module} in {end_time - start_time:.2f} seconds"
                            )
                            progress.update(task, advance=1)
                            continue

                    graph = self.graph.extract_graph(
                        code_path=dat["code_path"],
                        save_path=dat["graph"]["src_graph_path"],
                        overwrite=self.raw_overwrite,
                    )

                    num_nodes = len(graph["nodes"])
                    if not os.path.exists(dat["graph"]["node_feature_path"]):
                        node_feat = self.get_node_features(graph=graph)
                        assert node_feat.size(0) == num_nodes

                    all_mask = []
                    idx = 0
                    for i, tkey in enumerate(raw_data[key]["test_cases"].keys()):
                        if raw_data[key]["branches"][tkey] == []:
                            continue
                        if raw_data[key]["test_cases"][tkey]["code"] == "":
                            continue
                        try:
                            ast.parse(raw_data[key]["test_cases"][tkey]["code"])
                        except Exception as e:
                            continue
                        nkey = f"test_case_{idx}"
                        dat["test_cases"][nkey] = {}
                        dat["test_cases"][nkey]["test_case"] = raw_data[key][
                            "test_cases"
                        ][tkey]["code"]
                        dat["test_cases"][nkey]["branch"] = raw_data[key]["branches"][
                            tkey
                        ]

                        mask, branch_to_remove = self.get_mask_tensor(
                            graph=graph, branch=raw_data[key]["branches"][tkey]
                        )

                        if mask is None:
                            self.logger.log(
                                f"[red]Mask is empty for {key} test case {tkey}[/red]"
                            )
                            # mark branch to be removed
                            # key_to_remove.append(nkey)
                            continue

                        # self.logger.log(
                        #     f"Length of mask: {len(mask)}, length of branch: {len(dat['test_cases'][nkey]['branch'])}, removing branches: {branch_to_remove}"
                        # )

                        dat["test_cases"][nkey]["branch"] = [
                            b
                            for k, b in enumerate(raw_data[key]["branches"][tkey])
                            if k not in branch_to_remove
                        ]

                        assert len(dat["test_cases"][nkey]["branch"]) == len(
                            mask
                        ), "Mask and branch length mismatch: {} vs {}".format(
                            len(dat["test_cases"][nkey]["branch"]), len(mask)
                        )

                        all_mask.append(mask)
                        idx += 1

                    if len(all_mask) == 0:
                        self.logger.log(f"[red]No valid test cases for {key}[/red]")
                        continue

                    # for tkey in key_to_remove:
                    #     dat

                    self.logger.log(f"Generated masks for {key}: {len(all_mask)}")
                    torch.save(all_mask, dat["graph"]["mask_path"])
                    torch.save(node_feat, dat["graph"]["node_feature_path"])
                    data.append(dat)
                    repos.append(raw_data[key]["repo"])
                    num_module += 1
                    end_time = time.time()
                    self.logger.log(
                        f"Processed module {num_module} in {end_time - start_time:.2f} seconds"
                    )
                    progress.update(task, advance=1)

            data_dict[data_n] = {dat["uuid"]: dat for dat in data}

        self.data = data_dict
        with open(os.path.join(self.data_path, "data_processed.json"), "w") as f:
            json.dump(self.data, f)

        num_project = len(set(repos))
        self.logger.log("Processed raw data")
        self.logger.log(f"Number of projects: {num_project}")
        self.logger.log(f"Number of modules: {num_module}")
        return

import os
import json
import torch
import shutil
from typing import List
from rich.console import Console
from rich.progress import Progress
from data.core import Data
from graph.core import Graph
from transformers import PreTrainedModel, PreTrainedTokenizer

KEY_ID = "id"


class TestGenEval(Data):

    def __init__(
        self,
        logger: Console,
        path: str,
        graph: Graph,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        llm_tokenizer: PreTrainedTokenizer,
        debug: bool = False,
    ) -> None:
        self.name = "TestGenEval"
        self.data_path = os.path.join(path, self.name)
        self.debug = debug
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            self.data = None
        else:
            if os.path.exists(os.path.join(self.data_path, "data_processed.jsonl")):
                with open(
                    os.path.join(self.data_path, "data_processed.jsonl"), "r"
                ) as file:
                    self.data = [json.loads(l) for l in file.readlines()]
            else:
                if not os.path.exists(os.path.join(self.data_path, "data.jsonl")):
                    logger.log("data.jsonl not found, please crawl the data")
                else:
                    logger.log(
                        "Found data.jsonl file, but not processed. PLEASE RUN `process_raw`"
                    )

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
                "Found data_processed jsonl file, do not need to process raw data"
            )
            # load data
            with open(os.path.join(self.data_path, "data_processed.json"), "r") as file:
                self.data = json.load(file)
        else:
            process = True

        if not process:
            return

        if not os.path.exists(os.path.join(self.data_path, "data.jsonl")):
            raise FileNotFoundError("data.jsonl not found, please crawl the data")

        with open(os.path.join(self.data_path, "data.jsonl"), "r") as file:
            raw_data = [json.loads(l) for l in file.readlines()]

        data_dict = {task[KEY_ID]: task for task in raw_data}
        # make projects dir
        code_path = os.path.join(self.data_path, "codes")
        graph_path = os.path.join(self.data_path, "graphs")
        if os.path.exists(code_path):
            shutil.rmtree(code_path)
        if os.path.exists(graph_path):
            shutil.rmtree(graph_path)
        os.makedirs(code_path)
        os.makedirs(graph_path)
        data = []
        repos = []
        num_module = 0
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=len(data_dict))
            for i, key in enumerate(data_dict.keys()):
                dat = {}
                dat["test_cases"] = {}
                dat["graph"] = {}
                dat["uuid"] = i + 1
                dat["code_path"] = os.path.join(
                    code_path, f"{data_dict[key][KEY_ID]}.py"
                )
                dat["graph"]["src_graph_path"] = os.path.join(
                    graph_path, f"{data_dict[key][KEY_ID]}.json"
                )
                dat["graph"]["node_feature_path"] = os.path.join(
                    graph_path, f"{data_dict[key][KEY_ID]}.pt"
                )
                dat["graph"]["mask_path"] = os.path.join(
                    graph_path, f"{data_dict[key][KEY_ID]}_mask.pt"
                )
                with open(dat["code_path"], "w") as file:
                    file.write(data_dict[key]["code_src"])

                graph = self.graph.extract_graph(
                    code_path=dat["code_path"], graph_path=dat["graph_path"]
                )
                node_feat = self.get_node_features(graph=graph)
                all_mask = []
                idx = 0
                for i, tkey in enumerate(data_dict["test_cases"].keys()):
                    if data_dict["branches"][tkey] == []:
                        continue
                    nkey = f"test_case_{idx}"
                    dat["test_cases"][nkey] = {}
                    dat["test_cases"][nkey]["test_case"] = data_dict["test_cases"][tkey]
                    dat["test_cases"][nkey]["branch"] = data_dict["branches"][tkey]
                    mask = self.get_mask_tensor(
                        graph=graph, branch=data_dict["branches"][tkey]
                    )
                    all_mask.append(mask)
                    idx += 1
                torch.save(all_mask, dat["graph"]["mask_path"])
                torch.save(node_feat, dat["graph"]["node_feature_path"])
                data.append(dat)
                repos.append(data_dict[key]["repo"])
                num_module += 1
                progress.update(task, advance=1)

        self.data = {dat["uuid"]: dat for dat in data}
        with open(os.path.join(self.data_path, "data_processed.json"), "w") as f:
            json.dump(self.data, f)

        stat_info = {}
        num_project = len(set(repos))
        self.logger.log("Processed raw data")
        self.logger.log(f"Number of projects: {num_project}")
        self.logger.log(f"Number of modules: {num_module}")
        return

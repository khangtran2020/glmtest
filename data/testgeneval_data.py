import os
import json
import shutil
from data.testgeneval_pipeline.swebench_docker.constants import KEY_ID
from typing import List
from rich.console import Console
from rich.progress import Progress
from data.core import Data
from graph.core import Graph


class TestGenEval(Data):

    def __init__(
        self,
        logger: Console,
        path: str,
        graph: Graph,
        debug: bool = False,
    ) -> None:
        self.name = "TestGenEval"
        self.data_path = os.path.join(path, self.name)
        self.debug = debug
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        else:
            if os.path.exists(os.path.join(self.data_path, "data_processed.jsonl")):
                with open(
                    os.path.join(self.data_path, "data_processed.jsonl"), "r"
                ) as file:
                    self.data = [json.loads(l) for l in file.readlines()]
            else:
                logger.log("data_processed.jsonl not found, please crawl the data")

        super().__init__(
            name=self.name,
            path=path,
            logger=logger,
            graph=graph,
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
        if os.path.exists(os.path.join(self.data_path, "data_processed.jsonl")):
            self.logger.log(
                "Found data_processed jsonl file, do not need to process raw data"
            )
            # load data
            with open(
                os.path.join(self.data_path, "data_processed.jsonl"), "r"
            ) as file:
                self.data = [json.loads(l) for l in file.readlines()]
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
        project_path = os.path.join(self.data_path, "projects")
        graph_path = os.path.join(self.data_path, "graphs")
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
        if os.path.exists(graph_path):
            shutil.rmtree(graph_path)
        os.makedirs(project_path)
        os.makedirs(graph_path)
        data = []
        repos = []
        num_module = 0
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=len(data_dict))
            for i, key in enumerate(data_dict.keys()):
                dat = {}
                dat["uuid"] = i + 1
                dat["project"] = data_dict[key][KEY_ID]
                dat["code_path"] = os.path.join(
                    project_path, f"{data_dict[key][KEY_ID]}.py"
                )
                dat["graph_path"] = os.path.join(
                    graph_path, f"{data_dict[key][KEY_ID]}.json"
                )
                with open(dat["code_path"], "w") as file:
                    file.write(data_dict[key]["code_src"])

                self.graph.extract_graph(
                    code_path=dat["code_path"], graph_path=dat["graph_path"]
                )
                data.append(dat)
                repos.append(data_dict[key]["repo"])
                num_module += 1
                progress.update(task, advance=1)

        self.data = data
        with open(os.path.join(self.data_path, "data_processed.jsonl"), "w") as f:
            for item in self.data:
                f.write(json.dumps(item) + "\n")

        stat_info = {}
        num_project = len(set(repos))
        self.logger.log("Processed raw data")
        self.logger.log(f"Number of projects: {num_project}")
        self.logger.log(f"Number of modules: {num_module}")
        return

import os
import sys
import yaml
import json
import torch
import shutil
import subprocess
from typing import List
from rich.console import Console
from rich.progress import Progress
from data.core import Data
from graph.core import Graph
from utils.utils import check_package_exists_in_pypi
from transformers import AutoTokenizer, AutoModel


class OSSFuzz(Data):

    def __init__(
        self,
        logger: Console,
        path: str,
        run_time: int,
        docker_image: str,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        llm_tokenizer: AutoTokenizer,
        num_cpu: int,
        graph: Graph,
        debug: bool = False,
        test_gen: bool = False,
    ) -> None:
        if docker_image is None:
            raise ValueError("Docker image is not provided")
        self.name = "OSSFuzz"
        self.data_path = os.path.join(path, self.name)
        self.run_time = run_time
        self.docker_image = docker_image
        self.num_cpu = num_cpu
        self.debug = debug
        self.test_gen_flag = test_gen
        # check if data.json exist:
        if not os.path.exists(os.path.join(self.data_path, "data.json")):
            self.data = None
            self.stat_info = {}
        else:
            with open(os.path.join(self.data_path, "data.json"), "r") as file:
                self.data = json.load(file)
            with open(os.path.join(self.data_path, "stat_info.json"), "r") as file:
                self.stat_info = json.load(file)
        super().__init__(
            name=self.name,
            path=path,
            logger=logger,
            graph=graph,
            feat_model=model,
            feat_tokenizer=tokenizer,
            llm_tokenizer=llm_tokenizer,
            num_cpu=num_cpu,
            debug=debug,
        )

    def crawl(self) -> None:

        # check if dataset path exist
        if os.path.exists(self.data_path):
            self.logger.log(
                f"dataset path: {self.data_path} existed, please double-check"
            )
            sys.exit("PATH EXISTED")
        os.makedirs(self.data_path)

        # check if project path exist
        project_path = os.path.join(self.data_path, "projects")
        if os.path.exists(project_path):
            self.logger.log("project path existed, please double-check")
            sys.exit("PATH EXISTED")
        os.makedirs(project_path)

        # clone ossfuzz to data_path
        with self.logger.status("Cloning ossfuzz to data_path") as status:
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/google/oss-fuzz.git",
                    os.path.join(self.data_path, "oss-fuzz"),
                ]
            )
            if result.returncode != 0:
                self.logger.log("Error: OSSFuzz is not cloned")
                self.logger.log(result.stderr)
                sys.exit("CLONE ERROR")

            # delete .git
            for f in os.listdir(os.path.join(self.data_path, "oss-fuzz")):
                if ".git" in f:
                    if os.path.isdir(os.path.join(self.data_path, "oss-fuzz", f)):
                        shutil.rmtree(os.path.join(self.data_path, "oss-fuzz", f))
                    else:
                        os.remove(os.path.join(self.data_path, "oss-fuzz", f))

            # print(result.stdout + result.stderr)
            if result.returncode != 0:
                self.logger.log("Error: .git is not deleted")
                self.logger.log(result.stderr)
                sys.exit("DELETE ERROR")
            self.logger.log("Cloned ossfuzz to data_path")

        # get github links of all projects
        projects = os.listdir(
            os.path.join(os.path.join(self.data_path, "oss-fuzz"), "projects")
        )

        with Progress(console=self.logger) as progress:

            task = progress.add_task("Crawling projects", total=len(projects))

            for project in projects:
                oss_fuzz_project_path = os.path.join(
                    os.path.join(os.path.join(self.data_path, "oss-fuzz"), "projects"),
                    project,
                )
                # read yaml file to dict
                yaml_file_path = os.path.join(oss_fuzz_project_path, "project.yaml")
                with open(yaml_file_path, "r") as file:
                    project_yaml = yaml.safe_load(file)

                # check project language is python or not
                if (
                    "language" not in project_yaml.keys()
                    or "main_repo" not in project_yaml.keys()
                ):
                    self.logger.log(f"Project {project} is not python project")
                    progress.advance(task)
                    continue

                if project_yaml["language"] != "python":
                    self.logger.log(f"Project {project} is not python project")
                    progress.advance(task)
                    continue

                # get github link
                github_link = project_yaml["main_repo"]
                if "github.com" not in github_link:
                    self.logger.log(f"Project {project} is not github project")
                    progress.advance(task)
                    continue

                # create directory
                os.makedirs(os.path.join(project_path, project))

                # clone project to project_path
                subprocess.run(
                    [
                        "git",
                        "clone",
                        github_link,
                        os.path.join(project_path, project, project),
                    ]
                )
                self.logger.log(f"Cloned {project} to project_path")

                # delete .git
                for f in os.listdir(os.path.join(project_path, project, project)):
                    if ".git" in f:
                        if os.path.isdir(
                            os.path.join(project_path, project, project, f)
                        ):
                            shutil.rmtree(
                                os.path.join(project_path, project, project, f)
                            )
                        else:
                            os.remove(os.path.join(project_path, project, project, f))

                progress.advance(task)

        self.logger.log("Crawling completed")

    def process_raw(self) -> None:

        self.pre_process_raw()

        # check if prcessed data exists
        if not os.path.exists(os.path.join(self.data_path, "processed_data.json")):
            if self.test_gen_flag:
                self.process_test_gen()
            else:
                raise ValueError("Test generation is not enabled")
        else:
            # load data.json file
            with open(os.path.join(self.data_path, "processed_data.json"), "r") as file:
                data = json.load(file)

            data_final = []
            # repos = []
            num_module = 0
            with Progress() as progress:
                task = progress.add_task("[cyan]Processing...", total=len(data))
                for i, key in enumerate(data.keys()):
                    dat = {}
                    dat["test_cases"] = {}
                    dat["graph"] = {}
                    dat["uuid"] = i + 1
                    dat["code_path"] = data[key]["module_path"]
                    dat["graph"]["src_graph_path"] = data[key]["graph_path"]

                    graph_path = "/".join(data[key]["graph_path"].split("/")[:-1])
                    dat["graph"]["node_feature_path"] = os.path.join(
                        graph_path, f"node_feat_sample_{i}.pt"
                    )
                    dat["graph"]["mask_path"] = os.path.join(
                        graph_path, f"mask_sample_{i}.pt"
                    )
                    # with open(dat["code_path"], "w") as file:
                    #     file.write(data_dict[key]["code_src"])

                    # graph = self.graph.extract_graph(
                    #     code_path=dat["code_path"], graph_path=dat["graph_path"]
                    # )
                    with open(dat["graph"]["src_graph_path"], "r") as file:
                        graph = json.load(file)
                    node_feat = self.get_node_features(graph=graph)
                    all_mask = []
                    idx = 0
                    for i, tkey in enumerate(data[key]["test_cases"].keys()):
                        if len(data[key]["test_cases"]) == 0:
                            continue
                        nkey = f"test_case_{idx}"
                        dat["test_cases"][nkey] = {}
                        dat["test_cases"][nkey]["test_case"] = data[key]["test_cases"][
                            tkey
                        ]["test_path"]
                        dat["test_cases"][nkey]["branch"] = data[key]["test_cases"][
                            tkey
                        ]["branch"]
                        mask = self.get_mask_tensor(
                            graph=graph, branch=data[key]["test_cases"][tkey]["branch"]
                        )
                        all_mask.append(mask)
                        idx += 1
                    torch.save(all_mask, dat["graph"]["mask_path"])
                    torch.save(node_feat, dat["graph"]["node_feature_path"])
                    data_final.append(dat)
                    progress.update(task, advance=1)

            self.data = {dat["uuid"]: dat for dat in data}
            with open(os.path.join(self.data_path, "processed_data.json"), "w") as f:
                json.dump(self.data, f)

    def pre_process_raw(self) -> bool:

        process = False
        # check if the data has been processeds
        if os.path.exists(os.path.join(self.data_path, "data.json")):
            self.logger.log("Found data json file, do not need to process raw data")
            # load data
            with open(os.path.join(self.data_path, "data.json"), "r") as file:
                self.data = json.load(file)

            self.data = sorted(self.data, key=lambda x: x["num_modules"])

            # load stat_info
            with open(os.path.join(self.data_path, "stat_info.json"), "r") as file:
                self.stat_info = json.load(file)
        else:
            process = True

        if not process:
            return process

        # process data
        data = []
        project_path = os.path.join(self.data_path, "projects")

        for i, project in enumerate(os.listdir(project_path)):

            dat = {}
            dat["uuid"] = i + 1
            dat["project"] = project
            dat["project_path"] = os.path.join(project_path, project)
            modules = []
            modules_name = []
            modules_path = []
            current_project_path = os.path.join(project_path, project, project)
            for root, dirs, files in os.walk(current_project_path):
                for file in files:
                    if file.endswith(".py") and "__" not in file:
                        modules.append(
                            os.path.join(root, file)
                            .replace(
                                os.path.join(dat["project_path"], dat["project"]) + "/",
                                "",
                            )
                            .replace("/", ".")
                            .replace(".py", "")
                        )
                        modules_name.append(
                            os.path.join(root, file)
                            .replace(
                                os.path.join(dat["project_path"], dat["project"]) + "/",
                                "",
                            )
                            .replace("/", ".")
                            .replace(".py", "")
                            .replace(".", "_")
                        )
                        modules_path.append(os.path.abspath(os.path.join(root, file)))
            dat["modules"] = modules
            dat["module_path"] = modules_path
            dat["module_name"] = modules_name
            dat["num_modules"] = len(modules)
            if len(modules) == 0:
                continue
            data.append(dat)

        data = sorted(data, key=lambda x: x["num_modules"])
        self.data = data
        # create package.txt for each project
        for dat in self.data:
            self.create_package_txt(data=dat)
        self.clean_up()
        with open(os.path.join(self.data_path, "data.json"), "w") as f:
            json.dump(self.data, f)

        stat_info = {}
        num_project = len(self.data)
        num_modules = sum([dat["num_modules"] for dat in self.data])
        self.logger.log("Processed raw data")
        self.logger.log(f"Number of projects: {num_project}")
        self.logger.log(f"Number of modules: {num_modules}")

        for dat in self.data:
            stat_info[dat["project"]] = dat["num_modules"]

        # Save statistic information
        stat_info["num_project"] = num_project
        stat_info["num_modules"] = num_modules
        self.stat_info = stat_info

        with open(os.path.join(self.data_path, "stat_info.json"), "w") as f:
            json.dump(self.stat_info, f)
        return False

    def create_package_txt(self, data: dict) -> None:

        if check_package_exists_in_pypi(data["project"]):
            with open(os.path.join(data["project_path"], "package.txt"), "w") as file:
                file.write(data["project"])
            self.logger.log(f"Created package.txt for {data['project']}")
            return

        result = subprocess.run(
            ["pipreqs", "--force", f"{data['project_path']}"],
            capture_output=True,
            text=True,
        )
        if "error" in result.stderr.lower():
            self.logger.log(result.stderr)
            self.logger.log(
                f"Error: {data['project']} has error in creating requirements"
            )
        else:
            os.rename(
                os.path.join(data["project_path"], "requirements.txt"),
                os.path.join(data["project_path"], "package.txt"),
            )
            if check_package_exists_in_pypi(data["project"]):
                with open(
                    os.path.join(data["project_path"], "package.txt"), "r"
                ) as file:
                    packages = file.read()
                packages += f"\n{data['project']}"
                with open(
                    os.path.join(data["project_path"], "package.txt"), "w"
                ) as file:
                    file.write(packages)
                self.logger.log(f"Created package.txt for {data['project']}")
            self.logger.log(f"Created package.txt for {data['project']}")

    def clean_up(self) -> None:

        # Go over each project and if package.txt is not created, remove that project
        proj_to_remove = []
        for dat in self.data:
            if not os.path.exists(os.path.join(dat["project_path"], "package.txt")):
                shutil.rmtree(dat["project_path"])
                proj_to_remove.append(dat)
                self.logger.log(f"Removed {dat['project']}")

            # if number of modules is 0, remove that project
            if dat["num_modules"] == 0:
                shutil.rmtree(dat["project_path"])
                self.data.remove(dat)
                proj_to_remove.append(dat)
                self.logger.log(f"Removed {dat['project']}")

        for proj in proj_to_remove:
            self.data.remove(proj)
        self.logger.log("Cleaned up projects")

    def create_module_info(self) -> List[dict]:
        """
        Create a module info from the extracted data
        Each module info icnludes:
            - module_name_test_gen (e.g., path.to.module, without .py)
            - module_path
            - module_name
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
        module_infos = []
        for dat in self.data:
            project = dat["project"]
            package_path = dat["project_path"]
            for i, module in enumerate(dat["modules"]):
                module_info = {}
                module_info["module_name_full"] = f"{project}|{dat['module_name'][i]}"
                module_info["module_name"] = dat["module_name"][i]
                module_info["module_name_test_gen"] = module
                module_info["module_path"] = dat["module_path"][i]
                module_info["project"] = project
                module_info["project_path"] = package_path
                module_info["package_path"] = package_path
                module_info["code_path"] = os.path.join(package_path, project)
                module_info["output_test_path"] = os.path.join(package_path, "test")
                module_info["module_name_after_test_gen"] = (
                    f"test_{dat['module_name'][i]}.py"
                )
                module_info["graph_path"] = os.path.join(package_path, "graph")
                module_info["graph_name"] = f"{dat['module_name'][i]}.json"
                module_info["module_name_coverage"] = module.replace(".", "/")
                module_infos.append(module_info)
        return module_infos

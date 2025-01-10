import os
import sys
import yaml
import json
import select
import shutil
import pty
import subprocess
from typing import List
from rich.console import Console
from rich.progress import Progress
from data.core import Data
from graph.core import Graph
from utils.utils import check_package_exists_in_pypi


class Codamosa(Data):

    def __init__(
        self,
        logger: Console,
        path: str,
        run_time: int,
        docker_image: str,
        num_cpu: int,
        graph: Graph,
        debug: bool = False,
    ) -> None:
        if docker_image is None:
            raise ValueError("Docker image is not provided")
        self.name = "Codamosa"
        self.data_path = os.path.join(path, self.name)
        self.run_time = run_time
        self.docker_image = docker_image
        self.num_cpu = num_cpu
        self.debug = debug
        # check if data.json exist:
        if not os.path.exists(os.path.join(self.data_path, "data.json")):
            self.data = []
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
        with self.logger.status("Cloning codamosa to data_path") as status:
            result = subprocess.run(
                [
                    "git",
                    "lfs",
                    "clone",
                    "https://github.com/microsoft/codamosa.git",
                    os.path.join(self.data_path, "codamosa"),
                ]
            )
            if result.returncode != 0:
                self.logger.log("Error: Codamosa is not cloned")
                self.logger.log(result.stderr)
                sys.exit("CLONE ERROR")

            # delete .git
            for f in os.listdir(os.path.join(self.data_path, "codamosa")):
                if ".git" in f:
                    if os.path.isdir(os.path.join(self.data_path, "codamosa", f)):
                        shutil.rmtree(os.path.join(self.data_path, "codamosa", f))
                    else:
                        os.remove(os.path.join(self.data_path, "codamosa", f))

            # print(result.stdout + result.stderr)
            if result.returncode != 0:
                self.logger.log("Error: .git is not deleted")
                self.logger.log(result.stderr)
                sys.exit("DELETE ERROR")
            self.logger.log("Cloned codamosa to data_path")

        # load docker images for crawling modules
        images_paths = os.listdir(os.path.join(os.path.join(self.data_path, "codamosa"), "replication","docker-images"))
        for image in images_paths:
            image_path = os.path.join(os.path.join(self.data_path, "codamosa"), "replication","docker-images",image)
            result = subprocess.run(
                f"docker load < {image_path}",
                shell=True,
                check=True
            )

        # run docker images
        bash_file = os.path.join(os.path.join(self.data_path, "codamosa"), "replication", "scripts", "start_benchmark_container.sh")
        master, slave = pty.openpty()  # Open a pseudo-terminal
        try:
            # Start the subprocess
            process = subprocess.Popen(
                bash_file,
                shell=True,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                text=True
            )

            os.close(slave)  # Close the slave in the parent process

            # Read output from the master in a non-blocking manner
            while True:
                # Use select to wait for the master to have data to read
                ready, _, _ = select.select([master], [], [], 0.1)
                if ready:
                    output = os.read(master, 1024).decode()  # Read from master
                    print(output, end="")  # Print the output without adding extra newlines
                
                # Check if the process has terminated
                if process.poll() is not None:
                    break

            # Ensure all remaining output is read
            while True:
                ready, _, _ = select.select([master], [], [], 0.1)
                if ready:
                    output = os.read(master, 1024).decode()
                    if not output:
                        break
                    print(output, end="")
                else:
                    break

            print(f"\nScript exited with return code: {process.returncode}")

        finally:
            os.close(master)  # Close the master file descriptor


        
        # get github links of all projects
        # projects = os.listdir(
        #     os.path.join(os.path.join(self.data_path, "codamosa"), "replication","test-apps")
        # )

        projects = os.path.join(os.path.join(self.data_path, "codamosa"), "replication","test-apps")

        # move modules to projects
        shutil.copytree(src=projects, dst=project_path, dirs_exist_ok=True)
        shutil.rmtree(projects)

        self.logger.log("Crawling Codamosa completed")



    def process_raw(self) -> None:

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
            return

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
        return



    def create_package_txt(self, data: dict) -> None:
        #Codamosa project comes with package.txt already
        pass

    def clean_up(self) -> None:
        # Go over each project and if package.txt is not created, remove that project
        pass

    def create_module_info(self) -> List[dict]:
        """
        Create a module info from the extracted data
        Each module info includes:
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

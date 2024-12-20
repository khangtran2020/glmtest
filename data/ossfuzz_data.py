import os
import ast
import sys
import yaml
import json
import time
import shutil
import subprocess
from joblib import Parallel, delayed
from rich.console import Console
from rich.progress import Progress
from data.core import Data
from tqdm import tqdm
from graph.joerngraph import JoernGraph
from utils.utils import (
    check_package_exists_in_pypi,
    run_command,
    check_docker_image_exists,
)

# typing
from typing import List, Union

DOCKERFILE_TEMPLATE = """# Use nvidia/cuda image
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
WORKDIR /pynguin_gen

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda

RUN apt-get update
RUN apt-get install -y wget zip unzip bzip2 libc6-i386 libc6-x32 libfreetype6 \
    ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial \
    subversion vim libasound2 libxi6 libxtst6 && \
    apt-get clean

RUN wget https://download.oracle.com/java/19/archive/jdk-19.0.2_linux-x64_bin.deb && \
    dpkg -i jdk-19.0.2_linux-x64_bin.deb &&  \
    update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-19/bin/java 1 && \
    update-alternatives --install /usr/bin/javac javac /usr/lib/jvm/jdk-19/bin/javac 1 && \
    update-alternatives --config java &&  update-alternatives --config javac

        
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

RUN conda update conda \
    && conda create -n work python=3.10 -y

COPY ./ ./
RUN conda init bash
RUN echo "conda activate work" >> ~/.bashrc
RUN export PYTHONHASHSEED=0
ENV PATH /opt/conda/envs/pet/bin:$PATH
ENV CONDA_DEFAULT_ENV $work
"""

PYNGUIN_TEMPLATE = """docker run --rm -v {}:/input:ro -v {}:/output -v {}:/package:ro {} \
    --module-name {} --coverage_metrics BRANCH --maximum_search_time {} --report-dir /output --project_path /input --output-path /output --output_variables TargetModule,CoverageTimeline --assertion-generation NONE"""


class OSSFuzz(Data):

    def __init__(
        self,
        logger: Console,
        path: str,
        run_time: int,
        docker_image: str,
        num_cpu: int,
        graph: JoernGraph,
    ) -> None:
        if docker_image is None:
            raise ValueError("Docker image is not provided")
        self.name = "OSSFuzz"
        self.data_path = os.path.join(path, self.name)
        self.run_time = run_time
        self.docker_image = docker_image
        self.num_cpu = num_cpu
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
            name=self.name, path=path, logger=logger, graph=graph, num_cpu=num_cpu
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
        num_modules = [dat["num_modules"] for dat in self.data]
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

    def get_pynguin_command_for_module(self, module_info: dict) -> str:

        pynguin_command = PYNGUIN_TEMPLATE.format(
            os.path.abspath(module_info["code_path"]),
            os.path.abspath(module_info["output_path"]),
            os.path.abspath(module_info["project_path"]),
            self.docker_image,
            module_info["module_name"],
            self.run_time,
        )
        return pynguin_command

    def process_one_module(self, module_info) -> List[dict]:

        module_results_info = {}
        # gen test case with pynguin
        command = self.get_pynguin_command_for_module(module_info)
        run_command(command=command, capture_output=False)
        # check if test case is generated
        if not os.path.exists(
            os.path.join(
                module_info["output_path"], f"test_{module_info['module_name']}.py"
            )
        ):
            return []

        module_results_info["module_name"] = module_info["module_name"]
        module_results_info["module_path"] = module_info["module_path"]
        module_results_info["project"] = module_info["project"]

        # extract joern graph & locations
        self.graph.extract_graph(
            module_info["module_path"], save_path=f"{module_info['graph_path']}.json"
        )

        # if test case is generated, store the test case
        # count number of test cases
        test_path = os.path.join(
            module_info["output_path"], f"test_{module_info['module_name']}.py"
        )
        # check correct path
        assert os.path.exists(test_path)
        sub_test_path = os.path.join(
            module_info["output_path"], module_info["module_name"]
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
        return

    # def run_test_gen(self) -> None:

    #     # read data
    #     with open(os.path.join(self.data_path, "data.json"), "r") as file:
    #         self.data = json.load(file)

    #     # check if image exist
    #     if not check_docker_image_exists(self.docker_image):
    #         # build docker image for every project
    #         command = f"docker build -t {self.docker_image} -f pynguin/docker/Dockerfile --platform linux/amd64 ./pynguin"
    #         run_command(command=command, capture_output=False)
    #         self.logger.log(f"Built docker image for for every project")

    #     self.logger.log("Running test generation in parallel")
    #     commands_list = []
    #     for dat in self.data:
    #         commands_list += self.get_command_for_modules(dat)
    #     if self.num_cpu == -1:
    #         num_jobs = -1
    #         self.logger.log(
    #             f"Running test generation in parallel with {os.cpu_count()} cores"
    #         )
    #     else:
    #         if self.num_cpu > os.cpu_count():
    #             num_jobs = os.cpu_count()
    #             self.logger.log(
    #                 f"The indicated #CPUs is larger than the cores.\n"
    #                 + "Running test generation in parallel with {os.cpu_count()} cores"
    #             )
    #         else:
    #             num_jobs = self.num_cpu
    #             self.logger.log(
    #                 f"Running test generation in parallel with {self.num_cpu} cores"
    #             )
    #     results = Parallel(n_jobs=num_jobs)(
    #         delayed(run_command)(command=command, capture_output=False)
    #         for command in tqdm(commands_list)
    #     )
    #     self.logger.log("Test generation completed")
    #     return

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

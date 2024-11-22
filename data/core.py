import os
import json
import subprocess
from branch.extract import process_module
from rich.console import Console

# typing
from typing import Dict, List

dockerfile_template = """# Use nvidia/cuda image
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion vim && \
        apt-get clean
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
    && conda env create --name work

COPY ./ ./
RUN conda init bash
RUN echo "conda activate work" >> ~/.bashrc
RUN export PYTHONHASHSEED=0
ENV PATH /opt/conda/envs/pet/bin:$PATH
ENV CONDA_DEFAULT_ENV $work
"""

pynguin_template = """pynguin \
    --project-path {} \
    --output-path {} \
    --module-name {} --maximum-search-time 10 &"""

import requests


def check_package_exists(package_name: str) -> bool:

    url = f"https://pypi.org/pypi/{package_name}/json"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"The package '{package_name}' exists on PyPI.")
            return True
        else:
            print(f"The package '{package_name}' does NOT exist on PyPI.")
            return False
    except requests.RequestException as e:
        print(f"An error occurred while checking the package: {e}")
        return False


class Data(object):

    def __init__(self, name: str, original_name: str, data_path: str, logger: Console):
        self.name = name
        self.original_name = original_name
        self.data_path = data_path
        self.logger = logger
        self.dataset_path = os.path.abspath(os.path.join(self.data_path, self.name))
        self.project_path = os.path.join(self.dataset_path, "projects")
        self.logger.log(f"Dataset path: {self.dataset_path}")
        self.logger.log(f"Project path: {self.project_path}")

    def crawl(self):
        pass

    def process(self):
        pass

    def process_raw(self) -> None:
        """
        - Get all modules from project path
        - Create a dataframe to store all modules and their corresponding project with paths
        """

        # Check df exists
        if os.path.exists(os.path.join(self.dataset_path, "raw_data.json")):
            self.data = json.load(
                open(os.path.join(self.dataset_path, "raw_data.json"), "r")
            )
            self.logger.log("Found data json file, do not need to process raw data")
            return

        # Get all modules from project path
        data = []
        num_project = 0
        num_modules = 0

        for i, project in enumerate(os.listdir(self.project_path)):

            dat = {}
            dat["uuid"] = i + 1
            dat["project"] = project
            dat["project_path"] = os.path.join(self.project_path, project)
            dat["project_path_in_orignal"] = os.path.join(
                self.data_path, self.original_name, "projects", project
            )
            if "build.sh" in os.listdir(dat["project_path_in_orignal"]):
                dat["build_path"] = os.path.join(
                    dat["project_path_in_orignal"], "build.sh"
                )
            else:
                if check_package_exists(dat["project"]):
                    dat["build_path"] = "N/A"
                else:
                    continue
            num_project += 1
            modules = []
            project_path = os.path.join(self.project_path, project, project)
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    if file.endswith(".py") and "__" not in file:
                        modules.append(file)
            dat["modules"] = modules
            num_modules += len(modules)
            data.append(dat)

        # Create a json object and stor the data
        with open(os.path.join(self.dataset_path, "raw_data.json"), "w") as f:
            json.dump(data, f)

        self.data = data
        self.logger.log("Processed raw data")
        self.logger.log(f"Number of projects: {num_project}")
        self.logger.log(f"Number of modules: {num_modules}")

        # create Dockerfile for each project
        for dat in data:
            self._create_dockerfile(dat)
            self.logger.log(f"Created Dockerfile for {dat['project']}")
        return

    def run_pynguin(self, project: str = None) -> int:
        """
        - Create Dockerfile for each project
        - In each Dockerfile:
            - Install Anaconda and create a conda environment
            - Install all dependencies and Pynguin
            - Get list of modules and run Pynguin on all modules
        """
        pass

    def _run_pynguin_one_project(self, data: Dict) -> int:
        """
        - Run Pynguin on all modules of a project
        """
        pass

    def _create_dockerfile(self, data: Dict) -> None:
        """
        - Create a Dockerfile for a project
        """

        # read build.sh file
        if data["build_path"] == "N/A":
            if check_package_exists(data["project"]):
                new_build_sh = f"pip install {data['project']}"
            else:
                new_build_sh = f"pip install -r requirements.txt"
        else:
            with open(data["build_path"], "r") as file:
                build_sh = file.read()

            lines = build_sh.split("\n")
            new_lines = []
            for line in lines:
                if "# Build fuzzers into $OUT." in line:
                    break
                new_lines.append(line)
            new_build_sh = "\n".join(new_lines)
            with open(
                os.path.join(
                    data["project_path"], data["project"], "build_for_glmf.sh"
                ),
                "w",
            ) as file:
                file.write(new_build_sh)

        # Create bash script to install run pynguin
        modules = data["modules"]

        commands = []
        # run pynguin on all modules in parallel but only 10 at a time
        for i, module in enumerate(modules):
            pynguin_command = pynguin_template.format(
                data["project"],
                os.path.join("pynguin-results", data["project"], module),
                module,
            )
            commands.append(pynguin_command)
            if i % 10 == 0:
                commands.append("sleep 60")
        command = "\n".join(commands)
        with open(os.path.join(data["project_path"], "run_pynguin.sh"), "w") as file:
            file.write(command)

        # create Dockerfile
        # Add command to install dependencies by running build_for_glmf.sh
        dockerfile = (
            dockerfile_template
            + f"\nRUN cd {data['project']} && bash build_for_glmf.sh"
            + f"\nRUN bash run_pynguin.sh"
        )
        # write Dockerfile
        with open(os.path.join(data["project_path"], "Dockerfile"), "w") as file:
            file.write(dockerfile)

    def get_all_branches(self) -> None:
        """
        - Get all branches of a project
        """
        self.df["branches"] = self.df.apply(
            lambda x: process_module(x["path"], self.logger), axis=1
        )

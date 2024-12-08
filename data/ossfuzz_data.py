import os
import sys
import yaml
import json
import time
import shutil
import subprocess
from rich.console import Console
from rich.progress import Progress
from data.core import Data
from utils.utils import (
    check_package_exists_in_pypi,
    run_command,
    check_docker_image_exists,
)

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

PYNGUIN_TEMPLATE = """pynguin \
    --project-path {} \
    --output-path ./test/ \
    --module-name {} --maximum-search-time {} &"""

RUN_TEMPLATE = """#!/bin/bash
export PATH=$PATH:/home/pynguin_user/.local/bin
pip install pynguin coverage
export PYNGUIN_DANGER_AWARE=1


cd {}
bash build_for_glmf.sh

cd ..
"""


class OSSFuzz(Data):

    def __init__(self, logger: Console, path: str, run_time: int) -> None:
        self.name = "OSSFuzz"
        self.data_path = os.path.join(path, self.name)
        self.run_time = run_time
        super().__init__(name=self.name, path=path, logger=logger)

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

    def process(self) -> None:

        # check if the data has been processeds
        if os.path.exists(os.path.join(self.data_path, "data.json")):
            self.logger.log("Found data json file, do not need to process raw data")

            # load data
            with open(os.path.join(self.data_path, "data.json"), "r") as file:
                self.data = json.load(file)

            # load stat_info
            with open(os.path.join(self.data_path, "stat_info.json"), "r") as file:
                self.stat_info = json.load(file)

            # check dockerfile and build script
            for dat in self.data:
                if not os.path.exists(os.path.join(dat["project_path"], "Dockerfile")):
                    self.create_dockerfile(data=dat)
                    self.logger.log(f"Created Dockerfile for {dat['project']}")
                else:
                    self.logger.log(f"Found Dockerfile for {dat['project']}")

                if not os.path.exists(
                    os.path.join(
                        dat["project_path"], dat["project"], "build_for_glmf.sh"
                    )
                ):
                    self.create_build_script(data=dat)
                    self.logger.log(f"Created build script for {dat['project']}")
                else:
                    self.logger.log(f"Found build script for {dat['project']}")
            return

        # process data
        data = []
        stat_info = {}
        num_project = 0
        num_modules = 0
        project_path = os.path.join(self.data_path, "projects")

        for i, project in enumerate(os.listdir(project_path)):

            dat = {}
            dat["uuid"] = i + 1
            dat["project"] = project
            dat["project_path"] = os.path.join(project_path, project)
            dat["project_path_in_orignal"] = os.path.join(
                self.data_path, "oss-fuzz", "projects", project
            )
            if "build.sh" in os.listdir(dat["project_path_in_orignal"]):
                dat["build_path"] = os.path.join(
                    dat["project_path_in_orignal"], "build.sh"
                )
            else:

                if check_package_exists_in_pypi(dat["project"]):
                    dat["build_path"] = "N/A"
                else:
                    continue
            num_project += 1
            modules = []
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
            dat["modules"] = modules
            num_modules += len(modules)
            data.append(dat)
            stat_info[project] = len(modules)

        # Create a json object and stor the data
        with open(os.path.join(self.data_path, "data.json"), "w") as f:
            json.dump(data, f)

        self.data = data
        self.logger.log("Processed raw data")
        self.logger.log(f"Number of projects: {num_project}")
        self.logger.log(f"Number of modules: {num_modules}")

        # Save statistic information
        stat_info["num_project"] = num_project
        stat_info["num_modules"] = num_modules

        with open(os.path.join(self.data_path, "stat_info.json"), "w") as f:
            json.dump(stat_info, f)
        self.stat_info = stat_info

        # create Dockerfile for each project
        for dat in data:
            self.create_dockerfile(data=dat)
            self.create_build_script(data=dat)
            self.create_run_script(data=dat)

        # create dockerfile for all project
        with open(os.path.join(self.data_path, "Dockerfile"), "w") as f:
            f.write(DOCKERFILE_TEMPLATE)
        return

    def create_dockerfile(self, data: dict) -> None:
        with open(os.path.join(data["project_path"], "Dockerfile"), "w") as f:
            f.write(DOCKERFILE_TEMPLATE)
        self.logger.log(f"Created Dockerfile for {data['project']}")

    def create_build_script(self, data: dict) -> None:

        if check_package_exists_in_pypi(data["project"]):
            new_build_sh = f"pip install {data['project']}"
        else:
            with open(data["build_path"], "r") as file:
                build_sh = file.read()

            lines = build_sh.split("\n")
            new_lines = []
            for line in lines:
                if "fuzzer" in line.lower():
                    break
                new_lines.append(line)
            new_lines.append(f"pip install pynguin")
            new_build_sh = "\n".join(new_lines)
        with open(
            os.path.join(data["project_path"], data["project"], "build_for_glmf.sh"),
            "w",
        ) as file:
            file.write(new_build_sh)
        self.logger.log(f"Created build script for {data['project']}")

    def create_run_pynguin_script(self, data: dict) -> str:
        modules = data["modules"]

        project_template = "#!/bin/bash\n"
        # run pynguin on all modules in parallel but only 10 at a time
        for i, module in enumerate(modules):
            pynguin_command = PYNGUIN_TEMPLATE.format(
                f"./{data['project']}",
                module,
                self.run_time,
            )
            project_template += "\n" + pynguin_command
            if i + 1 % 10 == 0:
                self.logger.log(f"Checking sleeping time for {data['project']} at {i}")
                project_template += "sleep 60"
        with open(os.path.join(data["project_path"], "run_pynguin.sh"), "w") as file:
            file.write(project_template)
        self.logger.log(f"Created run pynguin script for {data['project']}")
        return project_template

    def create_run_script(self, data: dict) -> None:
        run_script = RUN_TEMPLATE.format(data["project"])
        run_script += "\n" + self.create_run_pynguin_script(data)
        with open(os.path.join(data["project_path"], "run.sh"), "w") as file:
            file.write(run_script)
        self.logger.log(f"Created run script for {data['project']}")

    def run_test_gen_one(self, data: dict) -> None:

        time_wait = len(data["modules"]) // 10 * 90
        self.logger.log(
            f"Running test generation for {data['project']} with {time_wait} seconds"
        )

        # run docker image
        container_name = f"{data['project']}"
        command = f"docker run -v {os.path.abspath(data['project_path'])}:/pynguin_gen --name {container_name} glmf bash run.sh"
        self.logger.log(f"Ran docker image for {data['project']}")
        self.logger.log("Running command: " + command)
        run_command(command=command, capture_output=False)

        # wait for test generation to complete
        self.logger.log(f"Waiting for {data['project']} to complete")
        time.sleep(time_wait)
        self.logger.log(f"Completed waiting for {data['project']}")

        # # copy results
        # command = (
        #     f"docker cp {container_name}:/pynguin_gen/test {data['project_path']}/test"
        # )
        # run_command(command=command, capture_output=False)
        # self.logger.log(f"Copied results for {data['project']}")

        # remove container
        command = f"docker rm -f {container_name}"
        run_command(command=command, capture_output=False)
        self.logger.log(f"Removed container for {data['project']}")

    def run_test_gen(self) -> None:

        # read data
        with open(os.path.join(self.data_path, "data.json"), "r") as file:
            self.data = json.load(file)

        # check if image exist
        if not check_docker_image_exists("glmf"):
            # build docker image for every project
            command = f"docker build -t glmf -f {os.path.join(self.data_path, 'Dockerfile')} {self.data_path}"
            run_command(command=command, capture_output=False)
            self.logger.log(f"Built docker image for for every project")

        with Progress(console=self.logger) as progress:
            task = progress.add_task("Running test generation", total=len(self.data))
            for dat in self.data:
                self.run_test_gen_one(data=dat)
                progress.advance(task)
                self.logger.log(f"Test generation completed for {dat['project']}")
        self.logger.log("Test generation completed")
        return

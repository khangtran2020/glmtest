import os
import sys
import yaml
import subprocess
import pandas as pd
from data.core import Data
from rich.console import Console
from rich.progress import Progress


class Ossfuzz(Data):

    def __init__(self, logger: Console, data_path: str) -> None:
        name = "OssFuzz"
        self.data_path = data_path
        super().__init__(name=name, logger=logger, data_path=data_path)

    def crawl(self) -> None:

        # check if dataset path exist
        if os.path.exists(self.dataset_path):
            self.logger.log(
                f"dataset path: {self.dataset_path} existed, please double-check"
            )
            sys.exit("PATH EXISTED")
        os.makedirs(self.dataset_path)

        # check if project path exist
        if os.path.exists(self.project_path):
            self.logger.log("project path existed, please double-check")
            sys.exit("PATH EXISTED")
        os.makedirs(self.project_path)

        # clone ossfuzz to dataset_path
        with self.logger.status("Cloning ossfuzz to dataset_path") as status:
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/google/oss-fuzz.git",
                    os.path.join(self.data_path, "oss-fuzz"),
                    "&&",
                    "rm",
                    "-rf",
                    "oss-fuzz/*.git",
                ]
            )
            if result.returncode != 0:
                self.logger.log("Error: OSSFuzz is not cloned")
                self.logger.log(result.stderr)
                sys.exit("CLONE ERROR")
            self.logger.log("Cloned ossfuzz to dataset_path")

        # get github links of all projects
        project_paths = os.path.join(
            os.path.join(self.data_path, "oss-fuzz"), "projects"
        )
        projects = os.listdir(project_paths)

        with Progress(console=self.logger) as progress:

            task = progress.add_task("Crawling projects", total=len(projects))

            for project in projects:
                project_path = os.path.join(project_paths, project)
                # read yaml file to dict
                yaml_file_path = os.path.join(project_path, "project.yaml")
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

                # clone project to project_path
                subprocess.run(
                    [
                        "git",
                        "clone",
                        github_link,
                        os.path.join(self.project_path, project),
                        "&&",
                        "rm",
                        "-rf",
                        f"{os.path.join(self.project_path, project)}/*.git",
                    ]
                )
                self.logger.log(f"Cloned {project} to project_path")
                progress.advance(task)

        self.logger.log("Crawling completed")
        self.process_raw()

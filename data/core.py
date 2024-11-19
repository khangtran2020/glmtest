import os
import subprocess
import pandas as pd
from branch.extract import process_module
from rich.console import Console


class Data(object):

    def __init__(self, name: str, logger: Console):
        self.name = name
        self.logger = logger
        self.dataset_path = os.path.abspath(os.path.join("../", self.name))
        self.project_path = os.path.join(self.dataset_path, "projects")

    def crawl(self):
        pass

    def process(self):
        pass

    def process_raw(self) -> None:
        """
        - Get all modules from project path
        - Create a dataframe to store all modules and their corresponding project with paths
        """
        # Get all modules from project path
        modules = []
        path = []
        for project in os.listdir(self.project_path):
            project_path = os.path.join(self.project_path, project)
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    if file.endswith(".py"):
                        modules.append(file)
                        path.append(os.path.join(root, file))

        # Create a dataframe to store all modules and their corresponding project with paths
        df = pd.DataFrame({"module": modules, "path": path})
        df["uuid"] = list(range(1, len(df) + 1))
        self.df = df
        self.logger.log("Processed raw data")

    def run_pynguin(self) -> int:
        """
        - Run Pynguin on all modules
        """

        # check docker image of pynguin
        image_name = "pynguin-docker"
        try:
            result = subprocess.run(
                ["docker", "images", "inspect", image_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode == 0:
                self.logger.log(
                    f"Pynguin docker image is found no ned to build the image"
                )
            else:
                self.logger.log(
                    "Pynguin docker image is not found. Need to build the image"
                )
                with self.logger.status("Building Pynguin docker image"):
                    try:
                        subprocess.run(
                            [
                                "docker",
                                "build",
                                "-t",
                                image_name,
                                "-f",
                                "docker/Dockerfile",
                                "--platform",
                                "linux/amd64",
                                ".",
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )

                        if result.returncode != 0:
                            self.logger.log(
                                f"Error: Pynguin docker image is not built with the following error: {result.stderr}"
                            )
                            return -1

                        self.logger.log("Pynguin docker image is built successfully")

                    except Exception as e:  # pragma: no cover
                        self.logger.log(f"Error: {e}")
                        return -1

        except Exception as e:
            self.logger.log(f"Error: {e}")
            return -1

        # run pynguin on all modules

    def get_all_branches(self) -> None:
        """
        - Get all branches of a project
        """
        self.df["branches"] = self.df.apply(
            lambda x: process_module(x["path"], self.logger), axis=1
        )

import os
import subprocess
import pandas as pd
from branch.extract import process_module
from rich.console import Console


def check_docker_image(image_name: str, logger: Console) -> int:
    try:
        result = subprocess.run(
            ["docker", "inspect", image_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if "No such object" in result.stdout:
            logger.log("Pynguin docker image is not found. Need to build the image")
            with logger.status("Building Pynguin docker image"):
                try:
                    subprocess.run(
                        [
                            "docker",
                            "build",
                            "-t",
                            image_name,
                            "-f",
                            "pynguin/docker/Dockerfile",
                            "--platform",
                            "linux/amd64",
                            "./pynguin",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )

                    if result.returncode != 0:
                        logger.log(
                            f"Error: Pynguin docker image is not built with the following error: {result.stderr}"
                        )
                        return -1

                    logger.log("Pynguin docker image is built successfully")

                except Exception as e:  # pragma: no cover
                    logger.log(f"Error: {e}")
                    return -1
        else:
            logger.log(f"Pynguin docker image is found no ned to build the image")
        logger.log("Pynguin docker image is ready")
        return 0
    except Exception as e:
        logger.log(f"Error: {e}")
        return -1


class Data(object):

    def __init__(self, name: str, data_path: str, logger: Console):
        self.name = name
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
        if os.path.exists(os.path.join(self.dataset_path, "raw_data.csv")):
            self.df = pd.read_csv(os.path.join(self.dataset_path, "raw_data.csv"))
            self.logger.log("Found csv file, do not need to process raw data")
            return

        # Get all modules from project path
        modules = []
        projects = []
        path = []
        for project in os.listdir(self.project_path):
            project_path = os.path.join(self.project_path, project)
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    if file.endswith(".py"):
                        modules.append(file)
                        projects.append(project)
                        path.append(os.path.join(root, file))

        # Create a dataframe to store all modules and their corresponding project with paths
        df = pd.DataFrame({"module": modules, "path": path, "project": projects})
        df["uuid"] = list(range(1, len(df) + 1))
        self.df = (
            df[["uuid", "project", "module", "path"]]
            .copy()
            .sample(frac=1)
            .reset_index(drop=True)
        )
        self.df.to_csv(os.path.join(self.dataset_path, "raw_data.csv"), index=False)
        self.logger.log(f"# of projects: {len(self.df['project'].unique())}")
        self.logger.log(f"# of modules: {len(self.df['module'].unique())}")
        self.logger.log("Processed raw data")
        return

    def run_pynguin(self) -> int:
        """
        - Run Pynguin on all modules
        """

        # check docker image of pynguin
        image_name = "pynguin-docker"
        result = check_docker_image(image_name=image_name, logger=self.logger)
        if result == -1:
            return -1

        # run pynguin on all modules

    def get_all_branches(self) -> None:
        """
        - Get all branches of a project
        """
        self.df["branches"] = self.df.apply(
            lambda x: process_module(x["path"], self.logger), axis=1
        )

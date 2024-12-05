import os
import json
import subprocess
from branch.extract import process_module
from rich.console import Console

# typing
from typing import List, Dict


class Data(object):

    def __init__(self, name: str, path: str, logger: Console) -> None:
        self.name = name  # name of the data
        self.path = path  # path of the raw data
        self.logger = logger

    def crawl(self) -> None:
        """
        Crawl the projects to the given path
        """
        pass

    def process(self) -> None:
        """
        - process the raw data to extract the modules and functions
        - create a self.data object to store the extracted data
        - save them to the given path in json format
        """
        pass

    def generate_testcase_pynguin(self) -> None:
        """
        Run Pynguin on the extracted modules and functions
        """
        pass

    def create_dockerfile(self) -> None:
        """
        Create a dockerfile to run pynguin
        on the extracted modules and functions
        """
        pass

    def run_test_gen(self) -> None:
        """
        Run the test generation process
        """
        pass

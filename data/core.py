import os
from rich.console import Console
from graph.core import Graph


class Data(object):

    def __init__(self, name: str, path: str, logger: Console, graph: Graph) -> None:
        self.name = name  # name of the data
        self.path = path  # path of the raw data
        self.logger = logger
        self.graph = graph

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

    def extract_graph(self) -> None:
        """
        Extract the graph from the raw data
        """
        # check if data is load to self.data
        if self.data is None:
            self.logger.log("Data not loaded, exiting...")
            return

        # extract graph from the data
        for dat in self.data:
            if os.path.exists(os.path.join(dat["project_path"], "graph")):
                self.logger.log(f"Graph already exists for {dat['project']}")
                continue
            self.graph.exporting_cpg(
                code_path=os.path.join(dat["project_path"], dat["project"]),
                save_path=os.path.join(dat["project_path"], "graph"),
            )

    def extract_locations(self) -> None:

        if self.data is None:
            self.logger.log("Data not loaded, exiting...")
            return

        # extract graph from the data
        for dat in self.data:
            if os.path.exists(os.path.join(dat["project_path"], "graph")):
                self.logger.log(f"Graph exists for {dat['project']}")
                self.graph.get_locations_and_id(
                    code_path=os.path.join(dat["project_path"], dat["project"]),
                    save_path=os.path.join(
                        dat["project_path"], "graph", "location.json"
                    ),
                    name=dat["project"],
                )

import os
from tqdm import tqdm
from rich.progress import Progress
from rich.console import Console
from graph.core import Graph


class Data(object):

    def __init__(
        self, name: str, path: str, logger: Console, graph: Graph, num_cpu: int
    ) -> None:
        self.name = name  # name of the data
        self.path = path  # path of the raw data
        self.logger = logger
        self.graph = graph
        self.num_cpu = num_cpu

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
        with Progress(console=self.logger) as progress:

            task = progress.add_task("Extracting graph", total=len(self.data))

            for dat in self.data:
                if os.path.exists(os.path.join(dat["project_path"], "graph")):
                    self.logger.log(f"Graph already exists for {dat['project']}")
                    continue

                os.makedirs(os.path.join(dat["project_path"], "graph"), exist_ok=False)

                sub_task = progress.add_task(
                    f"Exporting graphs for project {dat['project']}",
                    total=len(dat["module_path"]),
                )
                for i, module_path in enumerate(dat["module_path"]):
                    self.graph.exporting_cpg(
                        code_path=module_path,
                        save_path=os.path.join(
                            dat["project_path"], "graph", dat["module_name"][i]
                        ),
                    )
                    progress.advance(sub_task)
                progress.remove_task(sub_task)
                progress.advance(task)
            progress.remove_task(task)

    def extract_locations(self) -> None:

        if self.data is None:
            self.logger.log("Data not loaded, exiting...")
            return

        with Progress(console=self.logger) as progress:

            task = progress.add_task(
                "Extracting ids and locations", total=len(self.data)
            )

            for dat in self.data:

                sub_task = progress.add_task(
                    f"Crawling project {dat['project']}", total=len(dat["module_path"])
                )
                for i, module_path in enumerate(dat["module_path"]):
                    self.graph.get_locations_and_id(
                        code_path=module_path,
                        save_path=os.path.join(
                            dat["project_path"],
                            "graph",
                            dat["module_name"][i],
                            "location.json",
                        ),
                        name=dat["module_name"][i],
                    )
                    progress.advance(sub_task)

                progress.remove_task(sub_task)
                progress.advance(task)
            progress.remove_task(task)

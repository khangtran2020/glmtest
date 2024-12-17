import os
import json
import shutil
import subprocess
import nest_asyncio
from rich.console import Console
from graph.core import Graph
from graph.utils import extract_list_content, handle_location_out
from cpgqls_client import CPGQLSClient, import_code_query
from utils.utils import run_command, check_docker_image_exists

nest_asyncio.apply()


class JoernGraph(Graph):

    def __init__(
        self,
        host: str,
        port: str,
        joern_path: str,
        graph_path: str,
        docker_image: str,
        logger: Console,
    ) -> None:
        self.host = host
        self.port = port
        self.graph_path = graph_path
        self.joern_path = joern_path
        self.execution_path = os.path.join(self.joern_path, "joern-cli")
        self.docker_image = docker_image
        self.client = CPGQLSClient(f"{host}:{port}")
        super().__init__(logger)

    def import_code(self, code_path: str, name: str) -> None:
        query = import_code_query(os.path.abspath(code_path), name)
        result = self.client.execute(query)
        self.logger.log("Import code with result:" + result["stdout"])
        return

    def exporting_cpg(self, code_path: str, save_path: str) -> None:
        try:
            command = f"cd {self.execution_path} && ./joern-parse {os.path.abspath(code_path)}"
            run_command(command=command, capture_output=False)
            command = f"cd {self.execution_path} && ./joern-export --repr=all --format=dot --out {os.path.abspath(save_path)}"
            run_command(command=command, capture_output=False)
            shutil.rmtree(os.path.join(self.execution_path, "workspace"))
            self.logger.log(f"Exported CPG to {save_path}")
        except Exception as e:
            self.logger.log(f"Error exporting CPG: {e}")
        return

    def init_joern_server(self) -> None:

        if not check_docker_image_exists(self.docker_image):
            # build docker image for every project
            command = f"docker build -t {self.docker_image} -f graph/docker/Dockerfile --platform linux/amd64 ./graph"
            run_command(command=command, capture_output=False)
            self.logger.log(f"Built docker image for Joern")

        # run docker container
        command = f"docker run -d -p {self.port}:8080 --name joern {self.docker_image}"
        run_command(command=command, capture_output=False)
        self.logger.log(f"Started Joern server at {self.host}:{self.port}")
        return

    def install_joern_local(self) -> None:

        # check joern_path exists
        if not os.path.exists(self.joern_path):
            os.makedirs(self.joern_path)
            self.logger.log(f"Created directory {self.joern_path}")
        else:
            self.logger.log(f"Directory {self.joern_path} already exists")

        # download joern
        command = f"""curl -L "https://github.com/joernio/joern/releases/latest/download/joern-install.sh" -o {os.path.join(self.joern_path, 'joern-install.sh')}"""
        run_command(command=command, capture_output=False)
        self.logger.log(f"Downloaded Joern to {self.joern_path}")

        # install joern locally
        command = f"""chmod u+x {os.path.join(self.joern_path, 'joern-install.sh')} && {os.path.join(self.joern_path, 'joern-install.sh')} --install-dir={self.joern_path} --reinstall"""
        run_command(command=command, capture_output=False)
        self.logger.log("Installed Joern locally")
        return

    def get_locations_and_id(self, code_path: str, name: str, save_path: str) -> dict:

        self.import_code(code_path, name)
        # get id of all nodes
        query = """cpg.all.id.l"""
        result = self.client.execute(query)
        ids = [
            x.strip().replace("L", "")
            for x in extract_list_content(result["stdout"].replace("\n", " "))[0]
            .strip()
            .split(",")
        ]

        # get location of all nodes
        query = """cpg.all.location.l"""
        result = self.client.execute(query)
        data = handle_location_out(result["stdout"])

        # combine id and location
        locations = {}
        for i in range(len(ids)):
            locations[ids[i]] = data[i]
        self.logger.log(f"Got locations and id for {name}")

        # save locations to file
        with open(save_path, "w") as f:
            json.dump(locations, f)
        return locations

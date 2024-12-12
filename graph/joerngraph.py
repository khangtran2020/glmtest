import os
import subprocess
import nest_asyncio
from cpgqls_client import CPGQLSClient, import_code_query
from rich.console import Console
from utils.utils import run_command, check_docker_image_exists

nest_asyncio.apply()


class JoernGraph(object):

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
        self.logger = logger
        self.graph_path = graph_path
        self.joern_path = joern_path
        self.docker_image = docker_image
        self.client = CPGQLSClient(f"{host}:{port}")
        self.logger.print(f"Connected to Joern server at {host}:{port}")

    def import_code(self, code_path: str, name: str) -> None:
        query = import_code_query(os.path.abspath(code_path), name)
        result = self.client.execute(query)
        self.logger.log("Import code with result:" + result["stdout"])
        return

    def exporting_cpg(self, code_path: str) -> None:
        subprocess.run([os.path.join(self.joern_path, "joern-parse"), code_path])
        self.logger.log("Parsed code with Joern")
        subprocess.run(
            [
                os.path.join(self.joern_path, "joern-export"),
                "--repr=all",
                "--format=dot",
                "--out",
                "",
            ]
        )
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

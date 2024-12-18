import os
import json
import shutil
import subprocess
import nest_asyncio
from rich.console import Console
from graph.core import Graph
from cpgqls_client import CPGQLSClient, import_code_query
from utils.utils import (
    run_command,
    check_docker_image_exists,
    extract_list_content,
    handle_location_out,
)

nest_asyncio.apply()


class JoernGraph(Graph):

    def __init__(
        self,
        host: str,
        port: str,
        joern_path: str,
        docker_image: str,
        logger: Console,
    ) -> None:
        self.host = host
        self.port = port
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
            command = f"cd {self.execution_path} && ./joern-parse {os.path.abspath(code_path)} --language=PYTHONSRC"
            run_command(command=command, capture_output=False)
            command = f"cd {self.execution_path} && ./joern-export --repr=all --format=dot --out {os.path.abspath(save_path)}"
            run_command(command=command, capture_output=False)
            shutil.rmtree(os.path.join(self.execution_path, "workspace"))
            self.logger.log(f"Exported CPG to {save_path}")
        except Exception as e:
            self.logger.log(f"Error exporting CPG: {e}")
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

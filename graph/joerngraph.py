import os
import json
import nest_asyncio
from rich.console import Console
from graph.core import Graph
from cpgqls_client import CPGQLSClient, import_code_query

nest_asyncio.apply()


class JoernGraph(Graph):

    def __init__(
        self,
        port: str,
        joern_path: str,
        logger: Console,
    ) -> None:
        self.port = port
        self.joern_path = joern_path
        self.client = CPGQLSClient(f"localhost:{port}")
        super().__init__(logger)

    def import_code(self, code_path: str, name: str) -> None:
        query = import_code_query(os.path.abspath(code_path), name)
        result = self.client.execute(query)
        self.logger.log("Import code with result:" + result["stdout"])
        return

    def extract_graph(self, code_path: str, save_path: str, overwrite: bool) -> None:
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                graph = json.load(f)
        else:
            self.import_code(code_path, "work")
            graph = self.export_graph_data()
        if (not os.path.exists(save_path)) or (os.path.exists(save_path) and overwrite):
            self.save_to_json(graph, save_path)
        return graph

    def run_joern_query(self, query: str) -> str:
        try:
            result = self.client.execute(query)
            # print("result: ", result)
            stdout = result["stdout"]

            # Remove first line, and last two lines (the first and before last lines are just Scala specific output
            # that we don't need, and the last line is an empty line).
            stdout = stdout[stdout.find("\n") + 1 : stdout.rfind("\n")]
            stdout = "[\n" + stdout[: stdout.rfind("\n")] + "\n]"
            return stdout
        except Exception as e:
            return str(e)

    def export_graph_data(self) -> dict:
        # Export edges
        edges_command = 'cpg.graph.allEdges.map(e => Map("src" -> e.src.id, "dst" -> e.dst.id, "label" -> e.label, "id" -> e.hashCode)).l.toJsonPretty'
        edges_result = self.run_joern_query(edges_command)
        self.logger.log("Edges result:" + edges_result)
        edges = json.loads(edges_result)
        filtered_edges = []
        reachable_nodes = {}
        for edge in edges:
            edge["id"] = str(edge["id"])
            if edge["label"] in [
                "AST",
                "CFG",
                "CALL",
                "ARGUMENT",
                "RECEIVER",
                "CDG",
                "REACHING_DEF",
            ]:
                filtered_edges.append(edge)
                reachable_nodes[edge["src"]] = True
                reachable_nodes[edge["dst"]] = True

        # Export nodes
        nodes_command = 'cpg.all.map(n => Map("id" -> n.id, "label" -> n.label, "properties" -> n.properties, "location" -> n.location)).l.toJsonPretty'
        nodes_result = self.run_joern_query(nodes_command)
        self.logger.log("Nodes result:" + nodes_result)
        # print("nodes_result:", nodes_result)
        nodes = json.loads(nodes_result)
        filtered_nodes = []
        for node in nodes:
            if reachable_nodes.get(node["id"], False):
                filtered_nodes.append(node)
        graph = {"nodes": filtered_nodes, "edges": filtered_edges}
        return graph

    def save_to_json(self, data: dict, output_file: str) -> None:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Graph data saved to {output_file}")

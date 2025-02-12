import os
import dgl
import json
import torch
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from transformers import T5TokenizerFast, T5EncoderModel, AutoTokenizer
import torch
import argparse


class TrainDataset:
    def __init__(self, data_path, model, tokenizer):
        self.data_path = data_path
        self.data = self.process_data(data_path)
        self.model = model
        self.tokenizer = tokenizer
        self.type_of_graph = [
            "ARGUMENT",
            "RECEIVER",
            "CALL",
            "REACHING_DEF",
            "CDG",
            "CFG",
            "AST",
        ]

    def load_data(self, data_path):
        if data_path.endswith(".json"):
            with open(data_path, "r") as file:
                processed_data = json.load(file)
        elif data_path.endswith(".jsonl"):
            processed_data = []
            with open(data_path, "r") as file:
                for line in file:
                    processed_data.append(json.loads(line))
        else:
            raise ValueError("Unsupported file format. Please use .json or .jsonl.")
        return processed_data

    def get_graph_and_testcase(self, index: int):
        sub_data = self.data[index]
        graph = self.read_graph(os.path.join(sub_data["graph_path"]))
        test_cases = sub_data["test_cases"]
        return graph, test_cases

    def get_test(self, test_cases, index):
        return test_cases[list(test_cases.keys())[index]]

    def read_graph(self, graph_file):
        with open(graph_file, "r") as file:
            graph_dict = json.load(file)
        return graph_dict

    def get_num_nodes_from_raw(self, graph):
        return len(graph["nodes"])

    def get_mask(self, graph, test):
        mask = np.zeros(self.get_num_nodes_from_raw(graph))

        line_list = np.concatenate(np.array(test, dtype=object))
        for i in range(len(graph["nodes"])):
            node = graph["nodes"][i]
            if node["location"]["filename"] == "N/A":
                try:
                    if node["properties"]["LINE_NUMBER"] in line_list:
                        mask[i] = 1
                except:
                    mask[i] = 0

        return torch.Tensor([mask])

    def preprocess(self, graph):
        labels = []
        cnum = []
        order = []
        code = []
        lnum = []

        for node in graph["nodes"]:
            properties = node["properties"]
            labels.append(node["label"])
            try:
                cnum.append(properties["COLUMN_NUMBER"])
            except:
                cnum.append(-1)
            try:
                order.append(properties["ORDER"])
            except:
                order.append(-1)

            try:
                lnum.append(properties["LINE_NUMBER"])
            except:
                lnum.append(-1)

            try:
                if properties["CODE"] != "":
                    code.append(properties["CODE"])
                else:
                    code.append("EMPTY")
            except:
                code.append("EMPTY")

        nodes = pd.DataFrame(
            {
                "LABELS": np.array(labels),
                "COLUMN_NUMBER": np.array(cnum),
                "ORDER": np.array(order),
                "CODE": np.array(code),
                "LINE_NUMBER": np.array(lnum),
            }
        )

        return nodes

    def get_node_features(self, graph):
        df = self.preprocess(graph)
        embeddings = []
        labels = df["LABELS"]

        # Encode LABELS to integers
        label_encoder = LabelEncoder()
        df["LABELS_ENCODED"] = label_encoder.fit_transform(labels)

        # Get Code Embedding
        for code in df["CODE"].tolist():
            inputs = self.tokenizer(
                code, padding=True, truncation=True, return_tensors="pt", max_length=128
            ).to(self.model.device)
            with torch.no_grad():
                embedding = self.model.encoder(**inputs).last_hidden_state.mean(dim=1)[
                    0
                ]
            embeddings.append(embedding.to("cpu").numpy())

        df["CODE_FEATURE"] = embeddings
        # df = df.drop(["LABELS","CODE"],axis=1)
        c_features = deepcopy(df["CODE_FEATURE"])
        df = df[["LABELS_ENCODED", "COLUMN_NUMBER", "ORDER", "LINE_NUMBER"]]
        feat_df = torch.from_numpy(df.values).float()
        c_features = np.concatenate([np.expand_dims(e, 0) for e in c_features], axis=0)
        c_features = torch.from_numpy(c_features).float()
        feat = torch.cat([feat_df, c_features], dim=1)
        return feat

    def get_node_id_dict(self, graph):
        node_dict = {}
        for i in range(len(graph["nodes"])):
            node = graph["nodes"][i]
            node_dict[node["id"]] = i
        return node_dict

    def read_edge(self, graph):
        node_dict = self.get_node_id_dict(graph)
        edge_dict = {}
        for edge in graph["edges"]:
            if edge["label"] not in edge_dict:
                edge_dict[edge["label"]] = [
                    [node_dict[edge["src"]]],
                    [node_dict[edge["dst"]]],
                ]
            else:
                edge_dict[edge["label"]][0].append(node_dict[edge["src"]])
                edge_dict[edge["label"]][1].append(node_dict[edge["dst"]])
        return edge_dict

    def read_graphs(self, graph):
        graph_dict = {}
        num_nodes = self.get_num_nodes_from_raw(graph)
        feat = self.get_node_features(graph)

        assert num_nodes == feat.shape[0]

        edge_dict = self.read_edge(graph)

        for etype in edge_dict.keys():
            u = torch.Tensor(edge_dict[etype][0]).long()
            v = torch.Tensor(edge_dict[etype][1]).long()
            graph = dgl.graph((u, v), num_nodes=num_nodes)
            graph.ndata["feat"] = feat
            graph_dict[etype] = graph
        graph_dict["num_nodes"] = num_nodes
        graph_dict["feat_size"] = feat.size()
        return graph_dict

    def getInput(self, gIndex):
        file_path = self.data[gIndex]["module_path"]
        graph, _ = self.get_graph_and_testcase(gIndex)
        graph_dict = self.read_graphs(graph)
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        return file_content, graph_dict

    def processOut(self, content):
        template = "{} = <|fuzz|>{}<|/fuzz|>"
        res = ""
        test = content.split("\n")
        for out in test:
            if "=" in out:
                a = out.split("=")
                out = template.format(a[0], a[1])
            res += out + "\n"
        return res

    def get_prompt(self, file_content, output, tokenizer_Qwen):
        input = f"""\
        Generate the test case for the code below:
        ```
        {file_content}
        ```
        Here is the graph:
        ```
        <|graph_start|><|graph_pad|><|graph_end|>"
        ```
        """
        response = f"""\
        {output}
        """
        task_prompt = tokenizer_Qwen.apply_chat_template(
            [
                {"role": "user", "content": input},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        )
        return input, response, task_prompt

    def preprocess_pipeline(self, dataset_file, graph_file, mask_file, tokenizer_Qwen):
        graph_data = []
        mask_data = []
        with open(dataset_file, "a") as f:
            for i in range(len(self.data)):
                path = self.data[i]["graph_path"].split("graph/")
                test_path = os.path.join(path[0], "test", path[1].split(".json")[0])
                graph, testcases = self.get_graph_and_testcase(i)
                file_content, graph_dict = self.getInput(i)

                for j in range(len(testcases)):
                    print("I,J:", i, j)
                    specific_test_path = os.path.join(test_path, f"test_case_{j}.py")
                    # print(specific_test_path)
                    try:
                        with open(specific_test_path, "r", encoding="utf-8") as file:
                            output = file.read()
                        output = self.processOut(output)
                        # print(output)

                        test = self.get_test(testcases, j)
                        mask = self.get_mask(graph, test)
                        input, output, task_prompt = self.get_prompt(
                            file_content, output, tokenizer_Qwen
                        )
                        graphs = {
                            key: graph_dict[key]
                            for key in graph_dict
                            if isinstance(graph_dict[key], dgl.DGLGraph)
                        }

                        # Store metadata separately
                        metadata = {
                            "index": (i, j),
                            "input": input,
                            "response": output,
                            "task_prompt": task_prompt,
                        }

                        # Append graphs and metadata
                        graph_data.append(graphs)
                        mask_data.append(mask)
                        json.dump(metadata, f)

                        f.write("\n")
                    except:
                        continue

        # Save all graphs to a single file
        torch.save(graph_data, graph_file)
        torch.save(mask_data, mask_file)


def main():
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline for dataset generation."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./Dataset/OSSFuzz/processed_data.json",
        help="Path to the processed JSON data file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google-t5/t5-base",
        help="Pretrained model name or path for T5.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/project/phan/codellama/Testcase/",
        help="Cache directory for model and tokenizer files.",
    )
    parser.add_argument(
        "--codeqwen_path",
        type=str,
        default="../../CodeQwen1.5-7B-Chat",
        help="Path or model identifier for the CodeQwen tokenizer.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="./processed_data/datasets.jsonl",
        help="Output file path for the processed dataset (jsonl format).",
    )
    parser.add_argument(
        "--graph_file",
        type=str,
        default="./processed_data/graph.pt",
        help="Output file path for the graph data (pt format).",
    )
    parser.add_argument(
        "--mask_file",
        type=str,
        default="./processed_data/mask.pt",
        help="Output file path for the mask data (pt format).",
    )

    args = parser.parse_args()

    # Load the T5 tokenizer and model from the provided model name and cache directory.
    tokenizer = T5TokenizerFast.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, device_map="auto"
    )
    model = T5EncoderModel.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, device_map="auto"
    )

    # Load the tokenizer for CodeQwen.
    tokenizer_Qwen = AutoTokenizer.from_pretrained(
        args.codeqwen_path, device_map="auto"
    )

    data = TrainDataset(args.data_path, model, tokenizer)

    # Run the preprocessing pipeline with the provided output file paths and tokenizer.
    data.preprocess_pipeline(
        args.dataset_file, args.graph_file, args.mask_file, tokenizer_Qwen
    )


if __name__ == "__main__":
    main()

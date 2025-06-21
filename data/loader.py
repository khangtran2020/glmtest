import json
import torch
from data.utils import sampling_neighbor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any
from utils.constant import GRAPH_PAD_TOKEN


class GLMFDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 12000,
        baseline_prompt: str = "code",
        debug: bool = False,
        n_hops: int = 2,
        testing: bool = False,
        num_gpus: int = 1,
    ):
        self.data = data
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.baseline_prompt = baseline_prompt
        self.max_seq_length = max_seq_length
        self.graph_token_id = self.tokenizer.convert_tokens_to_ids([GRAPH_PAD_TOKEN])[0]
        self.debug = debug
        self.n_hops = n_hops
        self.testing = testing
        self.num_gpus = num_gpus
        self.index_to_key_dict = dict(zip(range(len(self.data)), self.data.keys()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(f"########### Loading sample {idx} from dataset")
        data_path = self.data[self.index_to_key_dict[idx]]
        with open(data_path, "r") as f:
            sample = json.load(f)
        graph_path = sample["graph_path"]
        graph = torch.load(graph_path)
        active_node = torch.Tensor(sample["active_node"])
        graph_mask = torch.Tensor(sample["mask"])
        # graph = sampling_neighbor(graph=graph, mask=active_node, n_hops=self.n_hops)

        for key in graph.keys():
            graph[key] = sampling_neighbor(
                graph=graph[key],
                mask=active_node,
                n_hops=self.n_hops,
            )

        if self.testing == False:
            # graph = sample["graph"]
            full_text = sample["full_text"]
            # graph_mask = sample["mask"]

            # Tokenize text input
            tokenized = self.tokenize(full_text)

            tokenized_user_prompt = self.tokenizer(sample["prompt"])
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized["labels"] = torch.cat(
                [
                    torch.Tensor([-100] * user_prompt_len).unsqueeze(0),
                    tokenized["labels"][:, user_prompt_len:],
                ],
                dim=1,
            ).long()
            input_ids = tokenized["input_ids"]

            if (
                self.baseline_prompt in ["graph", "graph_tr"]
                and self.graph_token_id not in input_ids
            ):
                raise ValueError("Input must contain graph token")

            return {
                "input": tokenized,
                "graph": graph,  # Should be a dictionary of graph structures
                "graph_mask": torch.tensor(graph_mask, dtype=torch.float),
            }
        else:
            # graph = sample["graph"]
            prompt = sample["prompt"]
            # graph_mask = sample["mask"]
            uuid = sample["uuid"]

            if self.num_gpus > 1:
                pass
            # Tokenize text input
            tokenized = self.tokenize(prompt, num_gpu=self.num_gpus)
            input_ids = tokenized["input_ids"]

            if (self.baseline_prompt in ["graph", "graph_tr"]) and (
                self.graph_token_id not in input_ids
            ):
                raise ValueError("Input must contain graph token")

            batch = {
                "input": tokenized,
                "graph": graph,  # Should be a dictionary of graph structures
                "graph_mask": torch.tensor(graph_mask, dtype=torch.float),
            }
            # print(uuid, batch)
            return (uuid, batch)

    def tokenize(
        self, prompt: str, add_eos_token: bool = True, num_gpu: int = 1
    ) -> dict:

        result = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )

        if num_gpu > 1:
            pad_tensor = (
                torch.tensor(
                    [self.tokenizer.pad_token_id]
                    * ((num_gpu * 2) - result["input_ids"].shape[1] % (num_gpu * 2))
                )
                .unsqueeze(0)
                .int()
                .to(result["input_ids"].device)
            )
            result["input_ids"] = torch.cat((pad_tensor, result["input_ids"]), dim=1)

            attention_tensor = torch.tensor(
                [[1] * pad_tensor.shape[1]],
                dtype=result["attention_mask"].dtype,
                device=result["attention_mask"].device,
            )

            result["attention_mask"] = torch.cat(
                [attention_tensor, result["attention_mask"]], dim=1
            )

        if result["input_ids"][0, -1] != self.tokenizer.eos_token_id and add_eos_token:

            # Create a tensor for the EOS token with shape (1, 1)
            eos_token_tensor = torch.tensor(
                [[self.tokenizer.eos_token_id]],
                dtype=result["input_ids"].dtype,
                device=result["input_ids"].device,
            )
            # Create a corresponding tensor for the attention mask
            attention_tensor = torch.tensor(
                [[1]],
                dtype=result["attention_mask"].dtype,
                device=result["attention_mask"].device,
            )

            # Concatenate along the sequence dimension (dim=1)
            result["input_ids"] = torch.cat(
                [result["input_ids"], eos_token_tensor], dim=1
            )
            result["attention_mask"] = torch.cat(
                [result["attention_mask"], attention_tensor], dim=1
            )

        # Use clone() to make a copy of the tensor for labels.
        result["labels"] = result["input_ids"].clone()

        return result


def collate_fn(batch) -> dict:

    # check if batch is tuple
    if not isinstance(batch[0], tuple):
        # print(batch)
        collated_input = {}
        for key in batch[0]["input"]:
            # Stack the tensors corresponding to the same key across the batch
            collated_input[key] = torch.stack(
                [sample["input"][key] for sample in batch]
            )
        collated = {
            "input": collated_input,
            # "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            # "labels": torch.stack([x["labels"] for x in batch]),
            "graph_mask": torch.stack([x["graph_mask"] for x in batch]),
            # Leave the graph as a list of dictionaries (or process as needed for your GNN)
            "graph": [x["graph"] for x in batch],
        }
        return collated
    else:
        print(batch)
        uuid, batch = batch
        collated_input = {}
        for key in batch[0]["input"]:
            # Stack the tensors corresponding to the same key across the batch
            collated_input[key] = torch.stack(
                [sample["input"][key] for sample in batch]
            )
        collated = {
            "input": collated_input,
            # "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            # "labels": torch.stack([x["labels"] for x in batch]),
            "graph_mask": torch.stack([x["graph_mask"] for x in batch]),
            # Leave the graph as a list of dictionaries (or process as needed for your GNN)
            "graph": [x["graph"] for x in batch],
        }
        return collated

import json
import torch
from rich import print as pprint
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
        dtype: str = "bf16",
        logger=None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.baseline_prompt = baseline_prompt
        self.max_seq_length = max_seq_length
        self.graph_token_id = self.tokenizer.convert_tokens_to_ids([GRAPH_PAD_TOKEN])[0]
        self.debug = debug
        self.n_hops = n_hops
        self.testing = testing
        self.num_gpus = num_gpus
        self.logger = logger
        self.dtype = dtype
        self.index_to_key_dict = dict(zip(range(len(self.data)), self.data.keys()))

        if self.logger is not None:
            self.logger.log(
                f"Dataset initialized with {len(self.data)} samples, max_seq_length={self.max_seq_length}, "
                f"baseline_prompt={self.baseline_prompt}, n_hops={self.n_hops}, testing={self.testing}, num_gpus={self.num_gpus}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import time

        start_time = time.time()

        data_path = self.data[self.index_to_key_dict[idx]]["path"]
        with open(data_path, "r") as f:
            sample = json.load(f)
        graph_path = sample["graph_path"]
        graph = torch.load(graph_path) if graph_path is not None else None

        if sample["active_node"] is not None:
            active_nodes = []
            for active_node in sample["active_node"]:
                active_nodes.append(torch.Tensor(active_node))
        else:
            active_nodes = None

        graph_masks = []
        if sample["mask"] is not None:
            for mask in sample["mask"]:
                graph_masks.append(torch.Tensor(mask))
        else:
            graph_masks = None

        if graph is not None:
            act_node = None
            for i, active_node in enumerate(active_nodes):
                if i == 0:
                    act_node = active_node
                else:
                    act_node = torch.cat((act_node, active_node), dim=0)

            for key in graph.keys():
                graph[key] = sampling_neighbor(
                    graph=graph[key],
                    mask=act_node,
                    n_hops=self.n_hops,
                )
                # Force features to CPU and materialize to avoid DGL lazy loading issues with DataLoader workers
                feat = graph[key].ndata["feat"]
                graph[key].ndata["feat"] = feat.to(
                    device="cpu",
                    dtype=torch.bfloat16 if self.dtype == "bf16" else torch.float16,
                )

        if self.testing == False:
            full_text = sample["full_text"]
            tokenized, pad_size = self.tokenize(full_text, num_gpu=self.num_gpus)

            tokenized_user_prompt = self.tokenizer(sample["prompt"])
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized["labels"] = torch.cat(
                [
                    torch.Tensor([-100] * (user_prompt_len + pad_size)).unsqueeze(0),
                    tokenized["labels"][:, (user_prompt_len + pad_size) :],
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
                "text": full_text,
                "input": tokenized,
                "graph": graph,  # Should be a dictionary of graph structures
                "graph_mask": (graph_masks if graph_masks is not None else None),
                "active_nodes": (active_nodes if active_nodes is not None else None),
                "data_load_time": time.time() - start_time,
            }
        else:
            prompt = sample["prompt"]
            uuid = sample["uuid"]

            tokenized, pad_size = self.tokenize(prompt, num_gpu=self.num_gpus)
            input_ids = tokenized["input_ids"]

            if (self.baseline_prompt in ["graph", "graph_tr"]) and (
                self.graph_token_id not in input_ids
            ):
                raise ValueError("Input must contain graph token")

            batch = {
                "text": prompt,
                "input": tokenized,
                "graph": graph,
                "graph_mask": (graph_masks if graph_masks is not None else None),
                "active_nodes": (active_nodes if active_nodes is not None else None),
                "data_load_time": time.time() - start_time,
            }
            return (uuid, batch)

    def tokenize(self, prompt: str, num_gpu: int = 1) -> dict:

        result = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        pad_size = 0
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

            result["input_ids"] = torch.cat(
                (pad_tensor, result["input_ids"]), dim=1
            )  # .to(dtype=torch.bfloat16)

            attention_tensor = torch.tensor(
                [[0] * pad_tensor.shape[1]],
                dtype=result["attention_mask"].dtype,
                device=result["attention_mask"].device,
            )

            result["attention_mask"] = torch.cat(
                [attention_tensor, result["attention_mask"]], dim=1
            ).to(dtype=torch.bfloat16)
            pad_size = pad_tensor.shape[1]

        # Use clone() to make a copy of the tensor for labels.
        result["labels"] = result["input_ids"].clone()
        return result, pad_size


def pad(
    input_tensors: List[torch.Tensor],
    pad_value: int,
    padding_side: str = "left",
) -> torch.Tensor:
    # Normalize tensors to 1D by squeezing once
    if input_tensors[0].dim() > 1:
        input_tensors = [tensor.squeeze(0) for tensor in input_tensors]

    # Get lengths using a single tensor operation
    lengths = torch.tensor([t.size(0) for t in input_tensors], dtype=torch.long)
    max_length = lengths.max().item()

    # Pre-allocate output tensor
    padded_tensors = torch.full(
        (len(input_tensors), max_length),
        pad_value,
        dtype=input_tensors[0].dtype,
        device=input_tensors[0].device,
    )

    if padding_side == "left":
        # Vectorized left padding using advanced indexing
        for i, (tensor, length) in enumerate(zip(input_tensors, lengths)):
            padded_tensors[i, max_length - length :] = tensor
    elif padding_side == "right":
        # Vectorized right padding using advanced indexing
        for i, (tensor, length) in enumerate(zip(input_tensors, lengths)):
            padded_tensors[i, :length] = tensor
    else:
        raise ValueError("padding_side must be 'left' or 'right'")

    return padded_tensors


def collate_fn(batch, tokenizer: PreTrainedTokenizer, max_seq_length: int) -> dict:

    # check if batch is tuple
    if not isinstance(batch[0], tuple):
        # print(batch)
        collated_input = {}

        input_ids = [sample["input"]["input_ids"] for sample in batch]
        attention_mask = [sample["input"]["attention_mask"] for sample in batch]
        labels = [sample["input"]["labels"] for sample in batch]

        collated_input["input_ids"] = pad(input_ids, pad_value=tokenizer.pad_token_id)
        collated_input["attention_mask"] = pad(attention_mask, pad_value=0)
        collated_input["labels"] = pad(labels, pad_value=-100).long()

        collated = {
            "text": [x["text"] for x in batch],
            "input": collated_input,
            "graph_mask": (
                [x["graph_mask"] for x in batch]
                if batch[0]["graph_mask"] is not None
                else None
            ),
            "graph": (
                [x["graph"] for x in batch] if batch[0]["graph"] is not None else None
            ),
            "data_load_time": [x.get("data_load_time", 0.0) for x in batch],
        }
        return collated
    else:
        uuid = [sample[0] for sample in batch]
        batch = [sample[1] for sample in batch]

        collated_input = {}

        input_ids = [sample["input"]["input_ids"] for sample in batch]
        attention_mask = [sample["input"]["attention_mask"] for sample in batch]
        labels = [sample["input"]["labels"] for sample in batch]

        collated_input["input_ids"] = pad(input_ids, pad_value=tokenizer.pad_token_id)
        collated_input["attention_mask"] = pad(attention_mask, pad_value=0)
        collated_input["labels"] = pad(labels, pad_value=-100).long()

        collated = {
            "text": [x["text"] for x in batch],
            "input": collated_input,
            "graph_mask": (
                [x["graph_mask"] for x in batch]
                if batch[0]["graph_mask"] is not None
                else None
            ),
            "graph": (
                [x["graph"] for x in batch] if batch[0]["graph"] is not None else None
            ),
            "data_load_time": [x.get("data_load_time", 0.0) for x in batch],
        }
        return uuid, collated

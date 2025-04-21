import torch
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
        testing: bool = False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.baseline_prompt = baseline_prompt
        self.max_seq_length = max_seq_length
        self.graph_token_id = self.tokenizer.convert_tokens_to_ids([GRAPH_PAD_TOKEN])[0]
        self.debug = debug
        self.testing = testing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.testing == False:
            sample = self.data[idx]
            graph = sample["graph"]
            full_text = sample["full_text"]
            graph_mask = sample["mask"]

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
            sample = self.data[idx]
            graph = sample["graph"]
            prompt = sample["prompt"]
            graph_mask = sample["mask"]
            uuid = sample["uuid"]

            # Tokenize text input
            tokenized = self.tokenize(prompt)
            input_ids = tokenized["input_ids"]

            if (
                self.baseline_prompt in ["graph", "graph_tr"]
                and self.graph_token_id not in input_ids
            ):
                raise ValueError("Input must contain graph token")

            batch = {
                "input": tokenized,
                "graph": graph,  # Should be a dictionary of graph structures
                "graph_mask": torch.tensor(graph_mask, dtype=torch.float),
            }
            # print(uuid, batch)
            return (uuid, batch)

    def tokenize(self, prompt: str, add_eos_token: bool = True) -> dict:

        result = self.tokenizer(
            prompt,
            return_tensors="pt",
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

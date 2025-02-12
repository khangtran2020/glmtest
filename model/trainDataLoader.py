import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

class CodeDataset(Dataset):
    def __init__(self, data,graph_list, graph_mask, tokenizer, max_seq_length=32000):
        self.data = data
        self.graph_list = graph_list
        self.graph_mask = graph_mask
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.graph_token_id = tokenizer.convert_tokens_to_ids(["<|graph|>"])[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Tokenize text input
        tokenized = self.tokenizer(
            sample["input"],
            # sample["task_prompt"],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        print(self.graph_token_id)
        # Add graph token to input_ids
        input_ids = tokenized["input_ids"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": self.tokenizer(
                sample["output"],
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )["input_ids"].squeeze(),
            "graph": self.graph_list[idx],  # Should be a dictionary of graph structures
            "graph_mask": torch.tensor(self.graph_mask[idx], dtype=torch.float)
        }
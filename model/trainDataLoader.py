import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

class CodeDataset(Dataset):
    def __init__(self, data,graph_list, graph_mask, tokenizer, max_seq_length=12000):
        self.data = data
        self.graph_list = graph_list
        self.graph_mask = graph_mask
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.graph_token_id = tokenizer.convert_tokens_to_ids(["<|graph_pad|>"])[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print("Idx:", idx)
        # print("graph: ", self.graph_list[idx])
        # print(len(self.graph_list))
        # print("masks: ", self.graph_mask)
        # print("masks Size: ", len(self.graph_mask[idx]))
        sample = self.data[idx]
        graph = self.graph_list[idx]
        
        prompt = sample["input"]+ "/n/n" + sample['response']
        
        # Tokenize text input
        tokenized = self.tokenize(
            prompt,
            # sample["task_prompt"],
            # max_length=self.max_seq_length,
            # padding="max_length",
            # truncation=True,
            # return_tensors="pt"
        )

        # print(tokenized["labels"].size())
        tokenized_user_prompt = self.tokenizer(sample["input"])
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        
        tokenized["labels"] = torch.cat([torch.Tensor([-100] * user_prompt_len).unsqueeze(0), 
        tokenized["labels"][
            :, user_prompt_len:
        ]], dim=1).long()
        
        
        
        
        # graph = self.graph_list[idx]
        # # for key in graph:
        # for key in graph.keys():
        #     graph[key].ndata['feat'] = graph[key].ndata['feat'].to(torch.float16)

        
        
        input_ids = tokenized["input_ids"]
        if self.graph_token_id in input_ids:
            has_graph = True
        else:
            raise ValueError("Input must contain graph token")
        
        return {
            "input": tokenized,
            # "attention_mask": tokenized["attention_mask"],
            # "labels": self.tokenizer(
            #     sample["response"],
            #     # max_length=self.max_seq_length,
            #     # padding="max_length",
            #     # truncation=True,
            #     return_tensors="pt"
            # )["input_ids"],
            # "graph": self.graph_list,  # Should be a dictionary of graph structures
            # "graph_mask": torch.tensor(self.graph_mask, dtype=torch.float)
            "graph": graph,  # Should be a dictionary of graph structures
            "graph_mask": torch.tensor(self.graph_mask[idx], dtype=torch.float)
        }

    def tokenize(self, prompt, add_eos_token=True):
         
        result = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        
        if (
            result["input_ids"][0, -1] != self.tokenizer.eos_token_id
            and add_eos_token
        ):
            
            # Create a tensor for the EOS token with shape (1, 1)
            eos_token_tensor = torch.tensor(
                [[self.tokenizer.eos_token_id]],
                dtype=result["input_ids"].dtype,
                device=result["input_ids"].device
            )
            # Create a corresponding tensor for the attention mask
            attention_tensor = torch.tensor(
                [[1]],
                dtype=result["attention_mask"].dtype,
                device=result["attention_mask"].device
            )
            
            # Concatenate along the sequence dimension (dim=1)
            result["input_ids"] = torch.cat([result["input_ids"], eos_token_tensor], dim=1)
            result["attention_mask"] = torch.cat([result["attention_mask"], attention_tensor], dim=1)
        
        # Use clone() to make a copy of the tensor for labels.
        result["labels"] = result["input_ids"].clone()
    
        return result
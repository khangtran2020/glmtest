import os
import copy
import torch
import transformers
from argparse import Namespace
from transformers import AdamW
import torch.distributed as dist
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from model.model import GLMFModelConfig, GLMFModelForCausalLM
from ring_flash_attn.zigzag_ring_flash_attn import zigzag_ring_flash_attn_func
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    Trainer,
)
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union


def get_train_dataloader(self) -> DataLoader:
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker

    return DataLoader(train_dataset, **dataloader_params)


Trainer.get_train_dataloader = get_train_dataloader


def extract_local(value, rank, world_size, device, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    return local_value.to(device)


def ring_flash_attention_forward(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
    seqlens_in_batch=None,
):
    attn_output = zigzag_ring_flash_attn_func(
        query_states,
        key_states,
        value_states,
        dropout,
        softmax_scale=softmax_scale,
        causal=self.is_causal,
    )
    return attn_output


transformers.modeling_flash_attention_utils._flash_attention_forward = (
    ring_flash_attention_forward
)

forward_qwen2_embed_ori = copy.deepcopy(Qwen2RotaryEmbedding.forward)


def forward_qwen2_embed(self, x, seq_len=None):
    seq_len = seq_len * dist.get_world_size()
    return forward_qwen2_embed_ori(self, x, seq_len)


Qwen2RotaryEmbedding.forward = forward_qwen2_embed


def judge_dir(resume_dir):
    is_checkpoint_dir = False
    if os.path.exists(resume_dir) == False:
        return False
    for _dir in os.listdir(resume_dir):
        if "checkpoint" in _dir:
            is_checkpoint_dir = True
        if "pth" in _dir:
            is_checkpoint_dir = True
    return is_checkpoint_dir


# IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "<s>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/CodeQwen1.5-7B-Chat")
    model_type: Optional[str] = field(default="llama")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    data_max_length: int = field(
        default=80000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={
            "help": "Additional trainable parameters except LoRA weights, if low rank training."
        },
    )
    resume_from_checkpoint: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    scaling_type: str = field(
        default="linear",
        metadata={"help": "Whether use flash attention for training."},
    )
    scaling_factor: int = field(
        default=1.0,
        metadata={"help": "Whether use flash attention for training."},
    )
    rope_theta: int = field(
        default=500000.0,
        metadata={"help": "Whether use flash attention for training."},
    )
    data_file: str = field(
        default="linear", metadata={"help": "Whether use flash attention for training."}
    )
    peft_model: str = field(
        default=None, metadata={"help": "Whether use flash attention for training."}
    )


def train_sp(args: Namespace):
    pass


def train(
    data_loader: torch.utils.data.DataLoader,
    dtype: str,
    use_lora: bool,
    llm_path: str,
    save_path: str,
    device: torch.device,
):
    ###Load Model
    tokenizer = AutoTokenizer.from_pretrained(llm_path, device_map="auto")
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|graph_start|>",
                "<|graph_pad|>",
                "<|graph_end|>",
                "<|fuzz|>",
                "<|/fuzz|>",
            ]
        }
    )

    config = GLMFModelConfig(
        llm_model=llm_path,
        use_lora=use_lora,
        dtype=dtype,
        device_map=device,
    )
    model = GLMFModelForCausalLM(config=config)
    print("###Done Loading Model and Tokenizer")

    # Training
    # Ensure model is on the correct device and in training mode.
    model.to(device)
    model.gnn.to("cpu")

    model.train()

    accumulation_steps = 4
    num_epochs = 3
    lr = 5e-5

    # Create the optimizer after applying the LoRA wrapper.
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    global_step = 0
    loss_track = []

    # Zero gradients initially.
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(data_loader):
            print("Batch step:", step)
            batch_loss = 0.0
            batch_size = batch["input"]["input_ids"].size(0)

            # Process each sample in the batch as a micro-batch.
            for i in range(batch_size):
                global_step += 1

                batch_input = batch["input"].copy()
                if "token_type_ids" in batch_input:
                    batch_input.pop("token_type_ids")

                micro_input = {
                    "input_ids": batch_input["input_ids"][i].to(device),
                    "attention_mask": batch_input["attention_mask"][i].to(device),
                    "labels": batch_input["labels"][i].to(device),
                }

                # Process the graph inputs. If they are tensors, move them to device.
                graph = batch["graph"][i]
                graph_mask = batch["graph_mask"][i]

                ##Change Device of graph
                # for key in graph.keys():
                #     graph[key]['feat'] = graph[key]['feat'].to(torch.bfloat16)

                # Change dtype
                # for g in graph.values():
                #     g.ndata['feat'] = g.ndata['feat'].to(torch.bfloat16)

                # Forward pass.

                outputs = model(
                    **micro_input,
                    graph=graph,
                    graph_mask=graph_mask,
                )
                # run_nvidia_smi()

                # del micro_input
                # del batch_input
                # gc.collect()
                # torch.cuda.empty_cache()

                # run_nvidia_smi()

                loss = outputs.loss
                loss = loss / accumulation_steps
                loss.backward()
                batch_loss += (
                    outputs.loss.item()
                )  # For logging (using the unscaled loss).

                # Update parameters once enough gradients have been accumulated.
                if global_step % accumulation_steps == 0:
                    optimizer.step()  # Update parameters.
                    optimizer.zero_grad()  # Reset gradients.

            # Log average loss for this batch.
            avg_batch_loss = batch_loss / batch_size
            loss_track.append(avg_batch_loss)
            print(f"Batch {step}: loss = {avg_batch_loss:.4f}")

    if model.config.use_lora == True:
        model.model = model.model.merge_and_unload()

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

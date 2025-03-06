import os
import copy
import torch
import types
import transformers
from argparse import Namespace
import torch.distributed as dist
from transformers import TrainingArguments
from transformers import Trainer, PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader
from data.loader import GLMFDataset, collate_fn
from transformers.trainer_utils import seed_worker
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from ring_flash_attn.zigzag_ring_flash_attn import zigzag_ring_flash_attn_func
from model.model import GLMFModelForCausalLM
from typing import Dict


def get_train_dataloader(self: Trainer) -> DataLoader:

    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_data = self.train_dataset
    data_collator = self.data_collator

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_data, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker

    data_loader = DataLoader(train_data, **dataloader_params)
    return data_loader


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


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(
        special_tokens_dict
    )  # minus 3 for the three special tokens due to graph tokens
    print("len(tokenizer)", len(tokenizer), num_new_tokens)
    num_new_tokens -= 3
    # model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def initialize_trainer_multi_gpu(
    args: Namespace,
    train_dataset: GLMFDataset,
    valid_dataset: GLMFDataset,
    model: GLMFModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    num_gpu: int,
) -> Trainer:

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        save_strategy="steps",
        save_steps=500,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        compute_metrics=None,
    )

    trainer.get_train_dataloader = types.MethodType(get_train_dataloader, trainer)

    if num_gpu > 1:
        transformers.modeling_flash_attention_utils._flash_attention_forward = (
            ring_flash_attention_forward
        )

        if model.model_type == "qwen2":
            forward_embed_ori = copy.deepcopy(Qwen2RotaryEmbedding.forward)
        elif model.model_type == "llama":
            forward_embed_ori = copy.deepcopy(LlamaRotaryEmbedding.forward)
        else:
            raise ValueError("Model type not supported")

        def forward_embed(self, x, seq_len=None):
            seq_len = seq_len * dist.get_world_size()
            return forward_embed_ori(self, x, seq_len)

        if model.model_type == "qwen2":
            Qwen2RotaryEmbedding.forward = forward_embed
        elif model.model_type == "llama":
            LlamaRotaryEmbedding.forward = forward_embed

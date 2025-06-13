import os
import re
import copy
import shutil
import torch
import subprocess
import transformers
import torch.distributed as dist
from utils.utils import seed_everything

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# from model.model import GLMFModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from ring_flash_attn.zigzag_ring_flash_attn import zigzag_ring_flash_attn_func
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from typing import Optional
from rich.console import Console

old_flash_attn = _flash_attention_forward


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


def patch_model(model_type: str, mode: str = "longlora"):
    if mode == "longlora":
        transformers.modeling_flash_attention_utils._flash_attention_forward = (
            longlora_flash_attention_forward
        )
    elif mode == "ring":
        transformers.modeling_flash_attention_utils._flash_attention_forward = (
            ring_flash_attention_forward
        )

        if model_type == "llama":
            forward_llama_embed_ori = copy.deepcopy(LlamaRotaryEmbedding.forward)

            def forward_llama_embed(self, x, seq_len=None):
                seq_len = seq_len * dist.get_world_size()
                return forward_llama_embed_ori(self, x, seq_len)

            Qwen2RotaryEmbedding.forward = forward_llama_embed

        elif model_type == "qwen2":

            forward_qwen2_embed_ori = copy.deepcopy(Qwen2RotaryEmbedding.forward)

            def forward_qwen2_embed(self, x, seq_len=None):
                seq_len = seq_len * dist.get_world_size()
                return forward_qwen2_embed_ori(self, x, seq_len)

            Qwen2RotaryEmbedding.forward = forward_qwen2_embed

        else:
            raise NotImplementedError(f"Model type {model_type} is not supported.")


def get_index_by_value(a, val):
    return (a == val).nonzero(as_tuple=True)[0]


def longlora_flash_attention_forward(
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

    bsz, q_len, _ = query_states.size()

    if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
        groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
        assert (
            q_len % groupsz == 0
        ), f"q_len {q_len} should be divisible by group size {groupsz}."
        num_groups = q_len // groupsz

        def shift(state: "torch.Tensor") -> "torch.Tensor":
            state = torch.cat(
                (
                    state[:, :, : self.num_heads // 2],
                    state[:, :, self.num_heads // 2 :].roll(-groupsz // 2, dims=1),
                ),
                dim=2,
            )
            return state.reshape(
                bsz * num_groups, groupsz, self.num_heads, self.head_dim
            )

        query_states, key_states, value_states = (
            shift(query_states),
            shift(key_states),
            shift(value_states),
        )
        if attention_mask is not None:
            attention_mask = attention_mask[:, :groupsz].repeat(num_groups, 1)

    attn_output: torch.Tensor = old_flash_attn(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_states.size(1),
        dropout=dropout,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    return attn_output


def run_nvidia_smi(console: Console):
    try:
        # Run the command and capture output
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # This makes stdout and stderr strings instead of bytes
        )
        if result.returncode == 0:
            if console is not None:
                console.log("nvidia-smi output:\n", result.stdout)
            else:
                print("nvidia-smi output:\n", result.stdout)
        else:
            if console is not None:
                console.log("Error executing nvidia-smi:\n", result.stderr)
            else:
                print("Error executing nvidia-smi:\n", result.stderr)
    except Exception as e:
        if console is not None:
            console.log("An exception occurred:", e)
        else:
            print("An exception occurred:", e)


def move_model_to_device(model, device):
    for name, param in model.named_parameters(recurse=True):
        if param.device.type == "meta":
            param = torch.nn.Parameter(torch.empty_like(param, device=device))
            setattr(model, name, param)
    for name, buffer in model.named_buffers(recurse=True):
        if buffer.device.type == "meta":
            buffer = torch.empty_like(buffer, device=device)
            setattr(model, name, buffer)
    return model


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    global_step: int,
    max_num_checkpoint: int,
    seed: int,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    while len(os.listdir(path)) >= max_num_checkpoint:
        oldest_checkpoint = min(
            [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.startswith("checkpoint-") and f.endswith(".pt")
            ],
            key=os.path.getctime,
        )
        shutil.rmtree(oldest_checkpoint)

    save_name = os.path.join(path, f"checkpoint-{global_step}.pt")

    checkpoint = {
        "seed": seed,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    # save checkpoint
    torch.save(checkpoint, save_name)


def load_checkpoint(
    path: str,
):
    if os.path.exists(path) == False:
        raise ValueError(f"Checkpoint path {path} does not exist.")
    checkpoint = torch.load(path)
    seed = checkpoint["seed"]
    seed_everything(seed)
    return checkpoint


def extract_code_block(markdown: str) -> Optional[str]:
    cleaned = re.sub(r"<\|/?fuzz\|>", "", markdown)

    pattern = r"```(?:\w+)?\n([\s\S]*?)```"
    match = re.search(pattern, cleaned)
    if match:
        return match.group(1)

    return cleaned

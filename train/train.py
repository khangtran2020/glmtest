import os
import gc
import json
import time
import torch
import wandb
import shutil
from rich import print as pprint
from functools import partial
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from datetime import timedelta
from data.core import Data
from data.loader import GLMFDataset, collate_fn
from model.model import (
    GLMFModelForCausalLM,
    get_model,
    continue_training_from_checkpoint,
)
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from accelerate import Accelerator
from accelerate import DeepSpeedPlugin
from transformers.trainer_utils import seed_worker
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
)
from train.utils import save_checkpoint
from inference.test import validate
from utils.utils import log_ram_usage
from collections import deque

# typing
import deepspeed
from argparse import Namespace
from rich.console import Console
from transformers import PreTrainedTokenizer


class StepTimer:
    """Simple timer for tracking average step time using a sliding window."""

    def __init__(self, window_size=100):
        self.step_times = deque(maxlen=window_size)
        self.start_time = None

    def start(self):
        """Start timing a step."""
        self.start_time = time.time()

    def end(self):
        """End timing a step and record the duration."""
        if self.start_time:
            self.step_times.append(time.time() - self.start_time)
            self.start_time = None

    def avg_time(self):
        """Get the average step time from the sliding window."""
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0.0

    def recent_time(self):
        """Get the most recent step time."""
        return self.step_times[-1] if self.step_times else 0.0


def train(
    args: Namespace,
    dataset: GLMFDataset,
    console: Console,
    continue_training: bool = False,
    max_num_checkpoint: int = 5,
    mixed_precision: str = "fp16",
    metadata: tuple = None,
    collate_fn: callable = collate_fn,
):
    # init accelerator
    if args.num_gpu == 1:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="wandb",
            project_dir=args.log_dir,
        )
    else:
        if args.use_deepspeed and os.path.exists(args.deepspeed_config):
            deepspeed_config_path = args.deepspeed_config
            if not os.path.exists(deepspeed_config_path):
                raise FileNotFoundError(
                    f"DeepSpeed config file not found at {deepspeed_config_path}."
                )

            # if ("gpt" in args.llm_model.lower()) or (mixed_precision == "bf16"):
            #     bf16_config = args.deepspeed_config.replace(".json", "_bf16.json")
            #     if not os.path.exists(bf16_config):
            #         raise FileNotFoundError(f"BF16 config not found at {bf16_config}.")
            #     deepspeed_config_path = bf16_config
            #     console.log(f"[green]Using BF16 config: {bf16_config}[/green]")

            with open(deepspeed_config_path, "r") as f:
                ds_config = json.load(f)

            zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
            zero3_init_flag = zero_stage == 3

            if zero3_init_flag:
                console.log(
                    f"[green]DeepSpeed ZeRO-3 detected - enabling zero3_init_flag for model initialization[/green]"
                )
            else:
                console.log(
                    f"[green]DeepSpeed ZeRO stage {zero_stage} detected[/green]"
                )

            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=deepspeed_config_path,
                zero3_init_flag=zero3_init_flag,
            )
            accelerator = Accelerator(
                gradient_accumulation_steps=args.batch_size,
                mixed_precision=mixed_precision,
                log_with="wandb",
                project_dir=args.log_dir,
                deepspeed_plugin=deepspeed_plugin,
            )
        else:
            accelerator = Accelerator(
                gradient_accumulation_steps=args.batch_size,
                mixed_precision=mixed_precision,
                log_with="wandb",
                project_dir=args.log_dir,
            )

    # Check if we're using ZeRO-3 for model initialization
    use_zero3 = False
    if args.use_deepspeed and os.path.exists(args.deepspeed_config):
        deepspeed_config_path = args.deepspeed_config
        if ("gpt" in args.llm_model.lower()) or (mixed_precision == "bf16"):
            bf16_config = args.deepspeed_config.replace(".json", "_bf16.json")
            if os.path.exists(bf16_config):
                deepspeed_config_path = bf16_config

        with open(deepspeed_config_path, "r") as f:
            ds_config = json.load(f)
        use_zero3 = ds_config.get("zero_optimization", {}).get("stage", 0) == 3

    model = get_model(
        args=args,
        console=console,
        tokenizer=dataset.llm_tokenizer,
        rank=accelerator.process_index,
        device=accelerator.device,
        metadata=metadata,
        use_zero3=use_zero3,
    )

    console.log(f"Metadata used for model initialization: {metadata}")
    if isinstance(model, deepspeed.runtime.engine.DeepSpeedEngine):
        console.log(f"[yellow]DeepSpeed model detected[/yellow]")

    if accelerator.is_main_process:
        for name, param in model.named_parameters():
            console.log(
                f"[yellow]Parameter {name}, dtype: {param.dtype}, shape: {param.shape}[/yellow]"
            )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-8)

    if args.continue_training:
        model, start_step, optimizer, lr_scheduler = continue_training_from_checkpoint(
            args=args,
            model=model,
            rank=accelerator.process_index,
            console=console,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
    else:
        start_step = -1

    collate_fn_ = partial(collate_fn, tokenizer=dataset.llm_tokenizer)
    if args.num_gpu == 1:
        train_single_gpu_accelerate(
            args=args,
            accelerator=accelerator,
            dataset=dataset,
            console=console,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            continue_training=continue_training,
            start_step=start_step,
            mixed_precision=mixed_precision,
            model=model,
            max_num_checkpoint=max_num_checkpoint,
            collate_fn=collate_fn_,
        )
    else:
        train_multi_gpu_accelerate(
            args=args,
            accelerator=accelerator,
            dataset=dataset,
            console=console,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            continue_training=continue_training,
            start_step=start_step,
            mixed_precision=mixed_precision,
            model=model,
            max_num_checkpoint=max_num_checkpoint,
            collate_fn=collate_fn_,
            use_zero3=use_zero3,
        )


def logging_train_data(
    console: Console, datasets: tuple, tokenizer: PreTrainedTokenizer
):

    tr_dataset, va_dataset = datasets
    console.log("Data prepared:")
    console.log(f"Train data: {len(tr_dataset)} data points")
    console.log(f"Valid data: {len(va_dataset)} data points")

    data_point_example = tr_dataset[0]
    console.log("Data prepared:")
    console.log(f"Train data: {len(tr_dataset)} data points")
    console.log(f"Valid data: {len(va_dataset)} data points")

    # print special token id of tokenizer and the associated ids
    console.log(
        f"[cyan]Tokenizer special tokens:[/cyan]\n{tokenizer.special_tokens_map}"
    )
    for key, value in tokenizer.special_tokens_map.items():
        if isinstance(value, str):
            value = tokenizer.convert_tokens_to_ids(value)
            console.log(f"[cyan]{key}[/cyan]: {value}")
        if isinstance(value, list):
            value = [tokenizer.convert_tokens_to_ids(v) for v in value]
            console.log(f"[cyan]{key}[/cyan]: {value}")

    console.log(
        f"[yellow]================ Example data point ================[/yellow]\n"
        + f"{data_point_example['text']}\n"
        + "[yellow]================ End of example data point ================[/yellow]"
    )
    len_of_tokenized = len(data_point_example["input"]["input_ids"].squeeze(0))
    len_of_labels = len(data_point_example["input"]["labels"].squeeze(0))
    len_of_attention_mask = len(
        data_point_example["input"]["attention_mask"].squeeze(0)
    )
    console.log(
        f"[yellow]================ Example tokenized - {len_of_tokenized} ================[/yellow]\n"
        + f"{data_point_example['input']['input_ids'].squeeze(0).tolist()}\n"
        + "[yellow]================ End of example tokenized ================[/yellow]"
    )
    console.log(
        f"[yellow]================ Example attention_mask - {len_of_attention_mask} ================[/yellow]\n"
        + f"{data_point_example['input']['attention_mask'].squeeze(0).tolist()}\n"
        + "[yellow]================ End of example tokenized ================[/yellow]"
    )
    console.log(
        f"[yellow]================ Example label - {len_of_labels} ================[/yellow]\n"
        + f"{data_point_example['input']['labels'].squeeze(0).tolist()}\n"
        + "[yellow]================ End of example label ================[/yellow]"
    )


def logging_gpu_usage(step: int, console: Console):
    gpu_memory = torch.cuda.memory_allocated() / (1024**3)
    gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
    gpu_free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    gpu_free = gpu_free / (1024**3)
    pprint(
        f"[blue]At step {step} - GPU memory allocated: {gpu_memory:.2f} GB, GPU memory reserved: {gpu_reserved:.2f} GB, GPU memory free: {gpu_free:.2f} GB[/blue]"
    )


def train_single_gpu_accelerate(
    args: Namespace,
    accelerator: Accelerator,
    dataset: Data,
    console: Console,
    model: GLMFModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    continue_training: bool = False,
    start_step: int = -1,
    max_num_checkpoint: int = 5,
    collate_fn: callable = collate_fn,
    mixed_precision: str = "bf16",
):

    # init wandb
    accelerator.init_trackers(
        project_name="GLMFuzz",
        config={
            "model_name": args.llm_model,
            "dataset": args.data,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size
            * args.gradient_accumulation_steps
            * accelerator.num_processes,
            "max_steps": args.num_train_epochs,
            "mixed_precision": mixed_precision,
            "seed": args.seed,
        },
        init_kwargs={"wandb": {"name": args.name}},
    )

    # Create save directory
    save_path = os.path.join(args.output_dir, args.name)
    console.log(f"Model will be saved to {save_path}...")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if os.path.exists(save_path):
        if continue_training == False:
            shutil.rmtree(save_path)
            os.makedirs(save_path, exist_ok=True)

    tokenizer = dataset.llm_tokenizer
    device = accelerator.device
    accelerator.print(f"Using {accelerator.num_processes} devices")
    accelerator.print(f"Mixed precision: {mixed_precision}")

    tr_dataset = GLMFDataset(
        data=dataset.train_data,
        tokenizer=dataset.llm_tokenizer,
        max_seq_length=args.max_seq_length,
        debug=args.debug,
        n_hops=dataset.n_hops,
        dtype=args.dtype,
        num_gpus=args.num_gpu,
        logger=console,
    )
    va_dataset = GLMFDataset(
        data=dataset.val_data,
        tokenizer=dataset.llm_tokenizer,
        max_seq_length=args.max_seq_length,
        debug=args.debug,
        n_hops=dataset.n_hops,
        dtype=args.dtype,
        num_gpus=args.num_gpu,
        logger=console,
    )
    tr_loader = DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    va_loader = DataLoader(
        va_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    logging_train_data(
        console=console, datasets=(tr_dataset, va_dataset), tokenizer=tokenizer
    )
    device = accelerator.device
    config = model.config

    # Prepare LLM for k-bit training if using quantization (CRITICAL for QLoRA)
    # Only affects model.llm_model, GNN remains unaffected
    if hasattr(config, "load_in_4bit") and (config.load_in_4bit or config.load_in_8bit):
        from peft import prepare_model_for_kbit_training

        model.llm_model = prepare_model_for_kbit_training(model.llm_model)
        console.log("[green]LLM prepared for k-bit training (QLoRA)[/green]")

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    global_step = 0
    previous_checkpoint_step = -1
    best_val_loss = 10000.0

    if continue_training == False:
        optimizer.zero_grad(set_to_none=True)

    with Progress(
        SpinnerColumn(),  # Shows a spinner
        TextColumn(
            "[progress.description]{task.description}"
        ),  # Displays additional info
        BarColumn(),  # Displays a progress bar
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # Shows percentage
        transient=True,
    ) as progress:

        train_task = progress.add_task("Training...", total=args.num_train_epochs)
        for epoch in range(args.num_train_epochs):
            model.train()
            train_epoch_task = progress.add_task(
                f"Epoch {epoch + 1}/{args.num_train_epochs}",
                total=len(tr_loader),
            )

            epoch_loss = 0.0
            num_items = 0.0

            # Time tracking variables
            total_data_time = 0.0
            total_embedding_time = 0.0
            total_forward_time = 0.0
            total_backward_time = 0.0
            num_batches = 0

            for step, batch in enumerate(tr_loader):
                batch_start_time = time.time()

                if (continue_training == True) and (global_step <= start_step):
                    global_step += args.batch_size
                    ram_usage = log_ram_usage()
                    progress.update(
                        train_epoch_task,
                        advance=1,
                        description=f"Batch {step + 1}/{len(tr_loader)}: loss = N/A - RAM usage: {ram_usage:.1f} MB",
                    )
                    continue

                data_load_time = time.time() - batch_start_time
                total_data_time += data_load_time

                global_step += 1
                batch_loss = 0.0
                batch_size = batch["input"]["input_ids"].size(0)

                # batch_input = batch["input"].copy()
                if "token_type_ids" in batch["input"]:
                    batch["input"].pop("token_type_ids")

                micro_input = {
                    "input_ids": batch["input"]["input_ids"].to(
                        device, non_blocking=True
                    ),
                    "attention_mask": batch["input"]["attention_mask"].to(
                        device, non_blocking=True
                    ),
                    "labels": batch["input"]["labels"].to(device, non_blocking=True),
                }
                user_prompt_lens = batch.get("user_prompt_lens", None)

                if "graph" in args.baseline_prompt:
                    graphs = []
                    graph_masks = []
                    graph_token_indices = []

                    for i in range(batch_size):
                        graph = batch["graph"][i]
                        graph = graph.pin_memory()
                        graph = graph.to(device, non_blocking=True)

                        graph_mask = [mask for mask in batch["graph_mask"][i]]

                        graph_token_index = torch.where(
                            micro_input["input_ids"][i]
                            == model.config.graph_token_id[1]
                        )[0].tolist()
                        graphs.append(graph)
                        graph_masks.append(graph_mask)
                        graph_token_indices.append(graph_token_index)
                else:
                    graphs = None
                    graph_masks = None
                    graph_token_indices = None

                with accelerator.accumulate(model):
                    forward_start = time.time()
                    outputs = model(
                        **micro_input,
                        step=global_step,
                        graphs=graphs,
                        graph_masks=graph_masks,
                        graph_token_indices=graph_token_indices,
                    )
                    forward_time = time.time() - forward_start
                    total_forward_time += forward_time

                    loss = outputs.loss
                    backward_start = time.time()
                    accelerator.backward(loss)
                    backward_time = time.time() - backward_start
                    total_backward_time += backward_time

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                # logging_gpu_usage(step=global_step, console=console)

                batch_loss += loss.item()
                avg_batch_loss = batch_loss / batch_size
                ram_usage = log_ram_usage()
                num_batches += 1

                # Get embedding time from model if available
                embedding_time = getattr(
                    model.module if hasattr(model, "module") else model,
                    "_last_embedding_time",
                    0.0,
                )
                total_embedding_time += embedding_time

                batch_total_time = time.time() - batch_start_time
                progress.update(
                    train_epoch_task,
                    advance=1,
                    description=f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f} | Data: {data_load_time:.3f}s | Emb: {embedding_time:.3f}s | Fwd: {forward_time:.3f}s | Bwd: {backward_time:.3f}s | RAM: {ram_usage:.1f}MB",
                )
                accelerator.print(
                    f"Step {global_step} - Loss: {avg_batch_loss:.4f} | "
                    f"Data: {data_load_time:.3f}s | Emb: {embedding_time:.3f}s | "
                    f"Fwd: {forward_time:.3f}s | Bwd: {backward_time:.3f}s"
                )
                epoch_loss += avg_batch_loss * batch_size
                num_items += batch_size

                for key in micro_input.keys():
                    micro_input[key] = micro_input[key].to("cpu")

                if "graph" in args.baseline_prompt:
                    for graph in graphs:
                        graph = graph.to("cpu")
                    for graph_mask in graph_masks:
                        for mask in graph_mask:
                            mask = mask.to("cpu")
                    del graph_masks, graphs
                outputs.logits = outputs.logits.to("cpu")
                loss = loss.to("cpu")
                del outputs, loss, micro_input
                gc.collect()
                torch.cuda.empty_cache()

                if global_step % args.logging_steps == 0:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    avg_data_time = (
                        total_data_time / num_batches if num_batches > 0 else 0
                    )
                    avg_embedding_time = (
                        total_embedding_time / num_batches if num_batches > 0 else 0
                    )
                    avg_forward_time = (
                        total_forward_time / num_batches if num_batches > 0 else 0
                    )
                    avg_backward_time = (
                        total_backward_time / num_batches if num_batches > 0 else 0
                    )

                    accelerator.log(
                        {
                            "train/loss": avg_batch_loss,
                            "train/learning_rate": current_lr,
                            "train/step": global_step,
                            "train/avg_data_time": avg_data_time,
                            "train/avg_embedding_time": avg_embedding_time,
                            "train/avg_forward_time": avg_forward_time,
                            "train/avg_backward_time": avg_backward_time,
                        },
                        step=global_step,
                    )
                    accelerator.print(
                        f"[cyan]Timing Stats (avg over {num_batches} batches): "
                        f"Data={avg_data_time:.3f}s | Emb={avg_embedding_time:.3f}s | "
                        f"Fwd={avg_forward_time:.3f}s | Bwd={avg_backward_time:.3f}s[/cyan]"
                    )

                if accelerator.sync_gradients and global_step % args.save_steps == 0:
                    if previous_checkpoint_step != -1:
                        old_dir = os.path.join(
                            save_path,
                            f"current_checkpoint",
                        )
                        new_dir = os.path.join(
                            save_path,
                            f"checkpoint-{previous_checkpoint_step}",
                        )
                        os.rename(old_dir, new_dir)
                    checkpoint_dir = os.path.join(
                        save_path,
                        f"current_checkpoint",
                    )
                    previous_checkpoint_step = global_step
                    save_checkpoint(
                        model=unwrapped_model,
                        path=checkpoint_dir,
                        global_step=global_step,
                        seed=args.seed,
                        is_lora=args.use_lora,
                    )
                    if accelerator.is_main_process:
                        accelerator.print(f"Saving checkpoint to {checkpoint_dir}")

                if (
                    accelerator.sync_gradients
                    and global_step % args.validating_steps == 0
                ):
                    console.log(
                        f"[green]Before validation: {torch.cuda.memory_allocated()}[/green]"
                    )
                    val_loss = validate(
                        args=args,
                        loader=va_loader,
                        model=model,
                        accelerator=accelerator,
                        config=config,
                        progress=progress,
                    )
                    wandb.log({"val_loss": val_loss})
                    console.log(
                        f"Validation loss: {val_loss:.4f} at step {global_step}"
                    )
                    if val_loss < best_val_loss:
                        # pass
                        best_val_loss = val_loss
                        console.log(
                            f"New best validation loss: {best_val_loss:.4f} at step {global_step}. Saving best model..."
                        )
                        checkpoint_dir = os.path.join(
                            save_path,
                            f"best_model",
                        )
                        if not os.path.exists(checkpoint_dir):
                            os.makedirs(checkpoint_dir, exist_ok=True)

                        model = model.to("cpu")
                        unwrapped_model = accelerator.unwrap_model(model).to("cpu")
                        torch.save(
                            unwrapped_model.state_dict(),
                            os.path.join(
                                checkpoint_dir, f"model_weight_step{global_step}.pt"
                            ),
                        )
                        tokenizer.save_pretrained(checkpoint_dir)
                        accelerator.print(f"Saving best checkpoint to {checkpoint_dir}")
                        model = model.to(device)

                        del unwrapped_model
                        del checkpoint_dir
                        gc.collect()

                    gc.collect()
                    torch.cuda.empty_cache()
                    console.log(
                        f"[blue]After validation: {torch.cuda.memory_allocated()}[/blue]"
                    )

            if ((continue_training == True) and (global_step > start_step)) or (
                continue_training == False
            ):
                progress.update(train_epoch_task, visible=False)
                progress.remove_task(train_epoch_task)
                progress.update(
                    train_task,
                    advance=1,
                    description=f"Epoch {epoch + 1}/{args.num_train_epochs}, loss = {epoch_loss / num_items:.4f}",
                )

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    # load best model for final evaluation
    best_model_path = os.path.join(save_path, "best_model")
    for file in os.listdir(best_model_path):
        if file.endswith(".pt"):
            best_model_path = os.path.join(best_model_path, file)
            break
    if os.path.exists(best_model_path):
        console.log(f"Loading best model from {best_model_path} for final evaluation")
        state_dict = torch.load(best_model_path, map_location="cpu")
        missing_keys, unexpected_keys = unwrapped_model.load_state_dict(
            state_dict, strict=False
        )
        if len(missing_keys) > 0:
            console.log(f"Missing keys when loading best model: {missing_keys}")
        if len(unexpected_keys) > 0:
            console.log(f"Unexpected keys when loading best model: {unexpected_keys}")
        console.log("Best model loaded successfully")

    if unwrapped_model.config.use_lora == True:
        unwrapped_model.llm_model = unwrapped_model.llm_model.merge_and_unload()
        unwrapped_model.config.use_lora = False
        console.log("[blue]Merged LoRA weights into the base model[/blue]")

    final_model_path = os.path.join(save_path, "final_model")
    console.log(f"Saving final model to {final_model_path}...")

    if not os.path.exists(final_model_path):
        os.makedirs(final_model_path, exist_ok=True)

    torch.save(
        unwrapped_model.state_dict(),
        os.path.join(final_model_path, "model_weight.pt"),
    )
    tokenizer.save_pretrained(final_model_path)

    console.log(f"Final model saved to {final_model_path}")
    # Log final model to W&B
    if wandb.run is not None:
        os.makedirs(os.path.join(save_path, "antifact"), exist_ok=True)
        model_artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description=f"Final model checkpoint",
        )
        model_artifact.add_dir(os.path.join(save_path, "antifact"))
        wandb.log_artifact(model_artifact)

    # End W&B run
    accelerator.end_training()


def train_multi_gpu_accelerate(
    args: Namespace,
    accelerator: Accelerator,
    dataset: Data,
    console: Console,
    model: GLMFModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    continue_training: bool = False,
    start_step: int = -1,
    max_num_checkpoint: int = 5,
    collate_fn: callable = collate_fn,
    mixed_precision: str = "bf16",
    use_zero3: bool = False,
):
    if accelerator.is_main_process:
        # init wandb
        accelerator.init_trackers(
            project_name="GLMFuzz",
            config={
                "model_name": args.llm_model,
                "dataset": args.data,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.batch_size,  # Micro-batching: process batch_size samples one-by-one
                "effective_batch_size": args.batch_size * accelerator.num_processes,
                "max_steps": args.num_train_epochs,
                "mixed_precision": mixed_precision,
                "seed": args.seed,
                "use_deepspeed": args.use_deepspeed,
            },
            init_kwargs={"wandb": {"name": args.name}},
        )

        # Create save directory
        save_path = os.path.join(args.output_dir, args.name)
        if os.path.exists(save_path):
            if continue_training == False:
                shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

    tokenizer = dataset.llm_tokenizer
    tr_dataset = GLMFDataset(
        data=dataset.train_data,
        tokenizer=dataset.llm_tokenizer,
        max_seq_length=args.max_seq_length,
        debug=args.debug,
        n_hops=dataset.n_hops,
        logger=console,
        dtype=args.dtype,
        metadata=model.metadata,
        num_gpus=args.num_gpu,
    )
    va_dataset = GLMFDataset(
        data=dataset.val_data,
        tokenizer=dataset.llm_tokenizer,
        max_seq_length=args.max_seq_length,
        debug=args.debug,
        n_hops=dataset.n_hops,
        logger=console,
        dtype=args.dtype,
        metadata=model.metadata,
        num_gpus=args.num_gpu,
    )

    repos = []
    for idx in range(len(tr_dataset)):
        uuid = list(tr_dataset.data.keys())[idx]
        repo = uuid.split("-")[0]
        repos.append(repo)

    repo_to_idx = {repo: idx for idx, repo in enumerate(set(repos))}
    repo_index_tensor = torch.tensor([repo_to_idx[repo] for repo in repos])
    repo_counts = torch.bincount(repo_index_tensor)
    weights = 1.0 / repo_counts.float()

    repo_weights = {repo: weights[idx].item() for repo, idx in repo_to_idx.items()}
    sample_weights = [repo_weights[repo] for repo in repos]

    # Log repo distribution
    if accelerator.is_main_process:
        unique_repos = len(set(repos))
        console.log(f"[cyan]Found {unique_repos} unique repos in training set[/cyan]")
        repo_dist = {
            list(repo_to_idx.keys())[i]: repo_counts[i].item()
            for i in range(len(repo_counts))
        }
        sorted_dist = dict(
            sorted(repo_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        console.log(f"[cyan]Top 10 repos by sample count: {sorted_dist}[/cyan]")
        console.log(
            f"[cyan]Using WeightedRandomSampler for balanced repo sampling[/cyan]"
        )

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(repos), replacement=True
    )

    dataloader_params = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "sampler": sampler,
        "num_workers": 1,  # Use os.cpu_count() workers
        "pin_memory": True,
        "prefetch_factor": 8,  # Increased from 2 to prefetch more batches
        "persistent_workers": True,
    }

    if not isinstance(tr_dataset, torch.utils.data.IterableDataset):
        dataloader_params["drop_last"] = True
        # Fix for transformers >= 4.46.0: seed_worker now requires (worker_id, num_workers, rank)
        dataloader_params["worker_init_fn"] = lambda worker_id: seed_worker(
            worker_id, num_workers=1, rank=accelerator.process_index
        )
        # Note: shuffle is not compatible with sampler, sampler handles sampling strategy

    tr_loader = DataLoader(tr_dataset, **dataloader_params)
    va_loader = DataLoader(
        va_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    config = model.config
    if hasattr(config, "load_in_4bit") and (config.load_in_4bit or config.load_in_8bit):
        from peft import prepare_model_for_kbit_training

        model.llm_model = prepare_model_for_kbit_training(model.llm_model)
        if accelerator.is_main_process:
            console.log("[green]LLM prepared for k-bit training (QLoRA)[/green]")

    model, optimizer, lr_scheduler, tr_loader, va_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, tr_loader, va_loader
    )

    global_step = 0
    previous_checkpoint_step = -1
    best_val_loss = 10000.0
    step_timer = StepTimer(window_size=100)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("| ⏱️  [cyan]{task.fields[step_time]:.2f}s/step"),
        transient=False,
    ) as progress:

        if accelerator.is_main_process:
            train_task = progress.add_task(
                "Training...", total=args.num_train_epochs, step_time=0.0
            )

        for epoch in range(args.num_train_epochs):

            model.train()
            if accelerator.is_main_process:
                train_epoch_task = progress.add_task(
                    f"Epoch {epoch + 1}/{args.num_train_epochs}",
                    total=len(tr_loader),
                    step_time=0.0,
                )
            epoch_loss = 0.0
            num_items = 0.0

            # Time tracking variables
            total_data_time = 0.0
            total_embedding_time = 0.0
            total_forward_time = 0.0
            total_backward_time = 0.0
            num_batches = 0

            for step, batch in enumerate(tr_loader):
                # Start step timing
                step_timer.start()

                if (continue_training == True) and (global_step <= start_step):
                    global_step += args.batch_size
                    ram_usage = log_ram_usage()

                    if accelerator.is_main_process:
                        progress.update(
                            train_epoch_task,
                            advance=1,
                            step_time=0.0,
                            description=f"Batch {step + 1}/{len(tr_loader)}: loss = N/A - RAM usage: {ram_usage:.1f} MB",
                        )
                    step_timer.end()
                    continue

                # Get data loading time from batch (measured in __getitem__)
                data_load_time = sum(batch.get("data_load_time", [0.0]))
                total_data_time += data_load_time

                accelerator.wait_for_everyone()
                batch_size = batch["input"]["input_ids"].size(0)
                batch_loss = 0.0

                # Micro-batching: Process each sample in batch individually to avoid OOM
                for micro_idx in range(batch_size):
                    # Extract single sample from batch
                    micro_input = {
                        "input_ids": batch["input"]["input_ids"][
                            micro_idx : micro_idx + 1
                        ],
                        "attention_mask": batch["input"]["attention_mask"][
                            micro_idx : micro_idx + 1
                        ],
                        "labels": batch["input"]["labels"][micro_idx : micro_idx + 1],
                    }

                    if "token_type_ids" in batch["input"]:
                        micro_input["token_type_ids"] = batch["input"][
                            "token_type_ids"
                        ][micro_idx : micro_idx + 1]

                    # Extract graph data for this sample
                    if "graph" in args.baseline_prompt:
                        micro_graphs = (
                            [batch["graph"][micro_idx]]
                            if batch["graph"] is not None
                            else None
                        )
                        micro_graph_masks = (
                            [batch["graph_mask"][micro_idx]]
                            if batch["graph_mask"] is not None
                            else None
                        )

                        if micro_graphs is not None:
                            graph_token_index = torch.where(
                                micro_input["input_ids"][0] == config.graph_token_id[1]
                            )[0].tolist()
                            micro_graph_token_indices = [graph_token_index]
                        else:
                            micro_graph_token_indices = None
                    else:
                        micro_graphs = None
                        micro_graph_masks = None
                        micro_graph_token_indices = None

                    # Forward pass with gradient accumulation
                    with accelerator.accumulate(model):
                        forward_start = time.time()
                        outputs = model(
                            **micro_input,
                            graphs=micro_graphs,
                            graph_masks=micro_graph_masks,
                            graph_token_indices=micro_graph_token_indices,
                            accelerator=accelerator,
                        )
                        forward_time = time.time() - forward_start
                        total_forward_time += forward_time

                        loss = outputs.loss
                        backward_start = time.time()
                        accelerator.backward(loss)
                        backward_time = time.time() - backward_start
                        total_backward_time += backward_time

                        if accelerator.sync_gradients:
                            accelerator.wait_for_everyone()
                            accelerator.clip_grad_norm_(
                                model.parameters(), args.max_grad_norm
                            )
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad(set_to_none=True)

                    # Accumulate loss
                    with torch.no_grad():
                        all_losses = accelerator.gather(loss)
                        all_losses = torch.where(
                            torch.isnan(all_losses),
                            torch.zeros_like(all_losses),
                            all_losses,
                        )
                        total_loss = torch.mean(all_losses)
                        batch_loss += total_loss.detach().float().item()

                    # Clean up memory for this micro-batch
                    for key in micro_input.keys():
                        micro_input[key] = micro_input[key].to("cpu")

                    if "graph" in args.baseline_prompt and micro_graphs is not None:
                        for graph in micro_graphs:
                            graph = graph.to("cpu")
                        if micro_graph_masks is not None:
                            for mask in micro_graph_masks:
                                for m in mask:
                                    m = m.to("cpu")
                        del micro_graphs, micro_graph_masks, micro_graph_token_indices

                    outputs.logits = outputs.logits.to("cpu")
                    loss = loss.to("cpu")
                    del outputs, loss, micro_input
                    gc.collect()
                    torch.cuda.empty_cache()

                # Update global step and logging
                global_step += 1

                # Clean up batch
                del batch
                gc.collect()

                # End step timing
                step_timer.end()

                avg_batch_loss = batch_loss / batch_size
                num_batches += 1

                # Get embedding time from model if available
                embedding_time = getattr(
                    model.module if hasattr(model, "module") else model,
                    "_last_embedding_time",
                    0.0,
                )
                total_embedding_time += embedding_time

                if accelerator.is_main_process:
                    ram_usage = log_ram_usage()
                    avg_time = step_timer.avg_time()
                    progress.update(
                        train_epoch_task,
                        advance=1,
                        step_time=avg_time,
                        description=f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f} | Data: {data_load_time:.3f}s | Emb: {embedding_time:.3f}s | Fwd: {forward_time:.3f}s | Bwd: {backward_time:.3f}s | RAM: {ram_usage:.1f}MB",
                    )
                    accelerator.print(
                        f"Step {global_step} - Loss: {avg_batch_loss:.4f} | "
                        f"Data: {data_load_time:.3f}s | Emb: {embedding_time:.3f}s | "
                        f"Fwd: {forward_time:.3f}s | Bwd: {backward_time:.3f}s"
                    )
                epoch_loss += avg_batch_loss * batch_size
                num_items += batch_size

                if (
                    global_step % args.logging_steps == 0
                ) and accelerator.is_main_process:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    avg_data_time = (
                        total_data_time / num_batches if num_batches > 0 else 0
                    )
                    avg_embedding_time = (
                        total_embedding_time / num_batches if num_batches > 0 else 0
                    )
                    avg_forward_time = (
                        total_forward_time / num_batches if num_batches > 0 else 0
                    )
                    avg_backward_time = (
                        total_backward_time / num_batches if num_batches > 0 else 0
                    )
                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "train/loss": avg_batch_loss,
                                "train/learning_rate": current_lr,
                                "train/step": global_step,
                                "train/avg_data_time": avg_data_time,
                                "train/avg_embedding_time": avg_embedding_time,
                                "train/avg_forward_time": avg_forward_time,
                                "train/avg_backward_time": avg_backward_time,
                            },
                            step=global_step,
                        )
                        accelerator.print(
                            f"[cyan]Timing Stats (avg over {num_batches} batches): "
                            f"Data={avg_data_time:.3f}s | Emb={avg_embedding_time:.3f}s | "
                            f"Fwd={avg_forward_time:.3f}s | Bwd={avg_backward_time:.3f}s[/cyan]"
                        )

                if global_step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    accelerator.print(f"Saving checkpoint at step {global_step}...")
                    if (previous_checkpoint_step != -1) and (
                        accelerator.is_main_process
                    ):

                        old_dir = os.path.join(
                            save_path,
                            f"current_checkpoint",
                        )
                        new_dir = os.path.join(
                            save_path,
                            f"checkpoint-{previous_checkpoint_step}",
                        )
                        os.rename(old_dir, new_dir)

                    checkpoint_dir_new = os.path.join(
                        save_path,
                        f"current_checkpoint",
                    )
                    previous_checkpoint_step = global_step

                    if accelerator.is_main_process:
                        while len(os.listdir(save_path)) >= max_num_checkpoint:
                            oldest_checkpoint = min(
                                [
                                    os.path.join(save_path, f)
                                    for f in os.listdir(save_path)
                                    if f.startswith("checkpoint-")
                                ],
                                key=os.path.getctime,
                            )
                            shutil.rmtree(oldest_checkpoint)

                    if use_zero3:
                        model.save_checkpoint(checkpoint_dir_new)
                        state_dict = get_fp32_state_dict_from_zero_checkpoint(
                            checkpoint_dir
                        )

                    unwrapped_model = accelerator.unwrap_model(model)
                    if accelerator.is_main_process:
                        save_checkpoint(
                            model=unwrapped_model,
                            path=checkpoint_dir_new,
                            global_step=global_step,
                            state_dict=state_dict if use_zero3 else None,
                            seed=args.seed,
                            is_lora=args.use_lora,
                            accelerator=accelerator,
                        )
                    accelerator.print(f"Saving checkpoint to {checkpoint_dir_new}")
                    del unwrapped_model

                if global_step % args.validating_steps == 0:
                    accelerator.wait_for_everyone()
                    val_loss = validate(
                        args=args,
                        loader=va_loader,
                        model=model,
                        config=config,
                        accelerator=accelerator,
                        progress=progress,
                    )
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        wandb.log({"val_loss": val_loss})
                        accelerator.print(
                            f"Validation loss: {val_loss:.4f} at step {global_step}"
                        )

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            accelerator.print(
                                f"New best validation loss: {best_val_loss:.4f} at step {global_step}. Saving best model..."
                            )
                            checkpoint_dir = os.path.join(
                                save_path,
                                f"best_model",
                            )

                            if not os.path.exists(checkpoint_dir):
                                os.makedirs(checkpoint_dir, exist_ok=True)

                            unwrapped_model = accelerator.unwrap_model(model)
                            # Save state dict to CPU to avoid keeping GPU memory
                            state_dict_cpu = {
                                k: v.cpu()
                                for k, v in unwrapped_model.state_dict().items()
                            }
                            if args.use_lora:
                                lora_state_dict = {
                                    k: v
                                    for k, v in state_dict_cpu.items()
                                    if ("lora_" in k) or ("gnn" in k) or ("nvib" in k)
                                }
                                torch.save(
                                    lora_state_dict,
                                    os.path.join(checkpoint_dir, f"model_weight.pt"),
                                )
                            else:
                                torch.save(
                                    state_dict_cpu,
                                    os.path.join(
                                        checkpoint_dir,
                                        f"model_weight.pt",
                                    ),
                                )

                            tokenizer.save_pretrained(checkpoint_dir)
                            console.log(
                                f"[green]Saved best checkpoint to {checkpoint_dir}[/green]"
                            )
                            del state_dict_cpu, unwrapped_model
                            del checkpoint_dir
                            if args.use_lora:
                                del lora_state_dict
                            gc.collect()

            if accelerator.is_main_process:
                if ((continue_training == True) and (global_step > start_step)) or (
                    continue_training == False
                ):
                    progress.update(train_epoch_task, visible=False)
                    progress.remove_task(train_epoch_task)
                    progress.update(
                        train_task,
                        advance=1,
                        description=f"Epoch {epoch + 1}/{args.num_train_epochs}, loss = {epoch_loss / num_items:.4f}",
                    )

    # One more validation at the end of training
    val_loss = validate(
        args=args,
        loader=va_loader,
        model=model,
        config=config,
        accelerator=accelerator,
        progress=None,
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.log({"val_loss": val_loss})
        accelerator.print(
            f"Final validation loss: {val_loss:.4f} at step {global_step}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            accelerator.print(
                f"New best validation loss: {best_val_loss:.4f} at step {global_step}. Saving best model..."
            )
            checkpoint_dir = os.path.join(
                save_path,
                f"best_model",
            )

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)

            unwrapped_model = accelerator.unwrap_model(model)
            # Save state dict to CPU to avoid keeping GPU memory
            state_dict_cpu = {
                k: v.cpu() for k, v in unwrapped_model.state_dict().items()
            }
            if args.use_lora:
                lora_state_dict = {
                    k: v
                    for k, v in state_dict_cpu.items()
                    if ("lora_" in k) or ("gnn" in k) or ("nvib" in k)
                }
                torch.save(
                    lora_state_dict,
                    os.path.join(checkpoint_dir, f"model_weight.pt"),
                )
            else:
                torch.save(
                    state_dict_cpu,
                    os.path.join(
                        checkpoint_dir,
                        f"model_weight.pt",
                    ),
                )

            tokenizer.save_pretrained(checkpoint_dir)
            console.log(f"[green]Saved best checkpoint to {checkpoint_dir}[/green]")
            del state_dict_cpu, unwrapped_model
            del checkpoint_dir
            if args.use_lora:
                del lora_state_dict
            gc.collect()

    accelerator.end_training()


"""def train_multi_gpu_accelerate_ring_attn(
    args: Namespace,
    dataset: Data,
    console: Console,
    model: GLMFModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    continue_training: bool = False,
    start_step: int = -1,
    max_num_checkpoint: int = 5,
    collate_fn: callable = collate_fn,
    mixed_precision: str = "bf16",
):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb",
        project_dir=args.log_dir,
    )

    process_group = dist.group.WORLD
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="GLMFuzz",
            config={
                "model_name": args.llm_model,
                "dataset": args.data,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.batch_size
                * args.gradient_accumulation_steps
                * accelerator.num_processes,
                "max_steps": args.num_train_epochs,
                "mixed_precision": mixed_precision,
                "seed": args.seed,
            },
            init_kwargs={"wandb": {"name": args.name}},
        )
        save_path = os.path.join(args.output_dir, args.name)
        if os.path.exists(save_path):
            if continue_training == False:
                shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
        console.log(f"Distributed type: {accelerator.distributed_type}")
        console.log(f"Number of processes: {accelerator.num_processes}")
        console.log(f"Mixed precision: {mixed_precision}")
    tokenizer = dataset.llm_tokenizer
    tr_dataset = GLMFDataset(
        data=dataset.train_data,
        tokenizer=dataset.llm_tokenizer,
        max_seq_length=args.max_seq_length,
        debug=args.debug,
        n_hops=dataset.n_hops,
        logger=console,
        dtype=args.dtype,
        num_gpus=args.num_gpu,
    )
    va_dataset = GLMFDataset(
        data=dataset.val_data,
        tokenizer=dataset.llm_tokenizer,
        max_seq_length=args.max_seq_length,
        debug=args.debug,
        n_hops=dataset.n_hops,
        logger=console,
        dtype=args.dtype,
        num_gpus=args.num_gpu,
    )
    dataloader_params = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    }

    if not isinstance(tr_dataset, torch.utils.data.IterableDataset):
        dataloader_params["drop_last"] = True
        # Fix for transformers >= 4.46.0: seed_worker now requires (worker_id, num_workers, rank)
        dataloader_params["worker_init_fn"] = lambda worker_id: seed_worker(
            worker_id, num_workers=4, rank=accelerator.process_index
        )

    tr_loader = DataLoader(tr_dataset, **dataloader_params)
    va_loader = DataLoader(
        va_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    if accelerator.is_main_process:
        logging_train_data(
            console=console, datasets=(tr_dataset, va_dataset), tokenizer=tokenizer
        )

    patch_model(process_group=process_group)
    device = accelerator.device
    config = model.config

    # Prepare LLM for k-bit training if using quantization (CRITICAL for QLoRA)
    # Only affects model.llm_model, GNN remains unaffected
    if hasattr(config, "load_in_4bit") and (config.load_in_4bit or config.load_in_8bit):
        from peft import prepare_model_for_kbit_training

        model.llm_model = prepare_model_for_kbit_training(model.llm_model)
        if accelerator.is_main_process:
            console.log("[green]LLM prepared for k-bit training (QLoRA)[/green]")

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # console.log(f"Model prepared with accelerator: {model}")
    global_step = 0
    previous_checkpoint_step = -1
    best_val_loss = 10000.0

    with Progress(
        SpinnerColumn(),  # Shows a spinner
        TextColumn(
            "[progress.description]{task.description}"
        ),  # Displays additional info
        BarColumn(),  # Displays a progress bar
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # Shows percentage
        transient=True,
    ) as progress:

        if accelerator.is_main_process:
            train_task = progress.add_task("Training...", total=args.num_train_epochs)

        for epoch in range(args.num_train_epochs):

            model.train()
            if accelerator.is_main_process:
                train_epoch_task = progress.add_task(
                    f"Epoch {epoch + 1}/{args.num_train_epochs}",
                    total=len(tr_loader),
                )
            epoch_loss = 0.0
            num_items = 0.0

            for step, batch in enumerate(tr_loader):

                if (continue_training == True) and (global_step <= start_step):

                    global_step += args.batch_size
                    ram_usage = log_ram_usage()

                    if accelerator.is_main_process:

                        progress.update(
                            train_epoch_task,
                            advance=1,
                            description=f"Batch {step + 1}/{len(tr_loader)}: loss = N/A - RAM usage: {ram_usage:.1f} MB",
                        )

                    continue

                accelerator.wait_for_everyone()
                global_step += args.batch_size
                batch_loss = 0.0
                batch_size = batch["input"]["input_ids"].size(0)

                accelerator.wait_for_everyone()

                if "token_type_ids" in batch["input"]:
                    batch["input"].pop("token_type_ids")

                micro_input = {
                    "input_ids": batch["input"]["input_ids"],
                    "attention_mask": batch["input"]["attention_mask"],
                    "labels": batch["input"]["labels"],
                }

                accelerator.wait_for_everyone()

                if "graph" in args.baseline_prompt:
                    graphs = batch["graph"]
                    graph_masks = batch["graph_mask"]
                    graph_token_indices = []

                    for i in range(batch_size):
                        graph_token_index = torch.where(
                            micro_input["input_ids"][i] == config.graph_token_id[1]
                        )[0].tolist()
                        graph_token_indices.append(graph_token_index)

                else:
                    graphs = None
                    graph_masks = None
                    graph_token_indices = None

                position_ids = (
                    torch.arange(micro_input["input_ids"].shape[1])
                    .unsqueeze(0)
                    .expand(micro_input["input_ids"].shape[0], -1)
                )

                accelerator.wait_for_everyone()

                with accelerator.accumulate(model):

                    outputs = model(
                        **micro_input,
                        position_ids=position_ids.to(device),
                        graphs=graphs,
                        graph_masks=graph_masks,
                        graph_token_indices=graph_token_indices,
                        step=global_step,
                        accelerator=accelerator,
                        ring_attn=True,
                    )

                    accelerator.wait_for_everyone()
                    loss = outputs.loss
                    accelerator.backward(loss)
                    # print(
                    #     f"Loss at rank {accelerator.process_index} - step {global_step}: {loss.item()}"
                    # )
                    accelerator.wait_for_everyone()

                if accelerator.sync_gradients:

                    accelerator.wait_for_everyone()
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    model.zero_grad(set_to_none=True)

                with torch.no_grad():
                    all_losses = accelerator.gather(loss)
                    # print(f"All losses at step {global_step}: {all_losses}")
                    all_losses = torch.where(
                        torch.isnan(all_losses),
                        torch.zeros_like(all_losses),
                        all_losses,
                    )
                    total_loss = torch.sum(all_losses)
                    batch_loss += total_loss.detach().float().item()

                for key in micro_input.keys():
                    micro_input[key] = micro_input[key].to("cpu")

                if "graph" in args.baseline_prompt:
                    for graph in graphs:
                        for key in GRAPH_KEYS:
                            if key in graph.keys():
                                graph[key] = graph[key].to("cpu")
                                graph.pop(key, None)
                    for graph_mask in graph_masks:
                        for mask in graph_mask:
                            mask = mask.to("cpu")
                    del graph_masks, graphs, graph_token_indices

                outputs.logits = outputs.logits.to("cpu")
                loss = loss.to("cpu")
                del outputs, loss, micro_input, batch
                gc.collect()
                torch.cuda.empty_cache()

                avg_batch_loss = batch_loss / batch_size
                if accelerator.is_main_process:
                    ram_usage = log_ram_usage()
                    # console.log(
                    #     f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f} - RAM usage: {ram_usage:.1f} MB"
                    # )
                    progress.update(
                        train_epoch_task,
                        advance=1,
                        description=f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.10f} - RAM usage: {ram_usage:.1f} MB",
                    )
                epoch_loss += avg_batch_loss * batch_size
                num_items += batch_size

                if (
                    global_step % args.logging_steps == 0
                ) and accelerator.is_main_process:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "train/loss": avg_batch_loss,
                                "train/learning_rate": current_lr,
                                "train/step": global_step,
                            },
                            step=global_step,
                        )
                        console.print(
                            f"[yellow]Step {global_step}: Loss: {avg_batch_loss:.4f}, LR: {current_lr:.6f}[/yellow]"
                        )

                if global_step % args.save_steps == 0:
                    accelerator.wait_for_everyone()

                    if (previous_checkpoint_step != -1) and (
                        accelerator.is_main_process
                    ):

                        old_dir = os.path.join(
                            save_path,
                            f"current_checkpoint",
                        )
                        new_dir = os.path.join(
                            save_path,
                            f"checkpoint-{previous_checkpoint_step}",
                        )
                        os.rename(old_dir, new_dir)

                    if accelerator.is_main_process:
                        checkpoint_dir_new = os.path.join(
                            save_path,
                            f"current_checkpoint",
                        )
                        previous_checkpoint_step = global_step

                        while len(os.listdir(save_path)) >= max_num_checkpoint:
                            oldest_checkpoint = min(
                                [
                                    os.path.join(save_path, f)
                                    for f in os.listdir(save_path)
                                    if f.startswith("checkpoint-")
                                ],
                                key=os.path.getctime,
                            )
                            # print(f"Removing oldest checkpoint: {oldest_checkpoint}")
                            shutil.rmtree(oldest_checkpoint)

                        unwrapped_model = accelerator.unwrap_model(model)
                        save_checkpoint(
                            model=unwrapped_model,
                            path=checkpoint_dir_new,
                            global_step=global_step,
                            seed=args.seed,
                            is_lora=args.use_lora,
                        )
                        accelerator.print(f"Saving checkpoint to {checkpoint_dir_new}")
                        del unwrapped_model

                if global_step % args.validating_steps == 0:
                    accelerator.wait_for_everyone()
                    val_loss = validate(
                        args=args,
                        loader=va_loader,
                        model=model,
                        config=config,
                        accelerator=accelerator,
                        progress=progress,
                    )
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        wandb.log({"val_loss": val_loss})
                        console.log(
                            f"Validation loss: {val_loss:.4f} at step {global_step}"
                        )

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            console.log(
                                f"New best validation loss: {best_val_loss:.4f} at step {global_step}. Saving best model..."
                            )
                            checkpoint_dir = os.path.join(
                                save_path,
                                f"best_model",
                            )

                            if not os.path.exists(checkpoint_dir):
                                os.makedirs(checkpoint_dir, exist_ok=True)

                            unwrapped_model = accelerator.unwrap_model(model)
                            torch.save(
                                unwrapped_model.state_dict(),
                                os.path.join(
                                    checkpoint_dir, f"model_weight_step{global_step}.pt"
                                ),
                            )
                            tokenizer.save_pretrained(checkpoint_dir)
                            console.log(
                                f"[green]Saved best checkpoint to {checkpoint_dir}[/green]"
                            )
                            del unwrapped_model
                            del checkpoint_dir
                            gc.collect()

                    # No need to restore dtypes - Accelerate handles mixed precision automatically
                    # Converting quantized 4-bit/8-bit weights to bfloat16 would break quantization
                    accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                if ((continue_training == True) and (global_step > start_step)) or (
                    continue_training == False
                ):
                    progress.update(train_epoch_task, visible=False)
                    progress.remove_task(train_epoch_task)
                    progress.update(
                        train_task,
                        advance=1,
                        description=f"Epoch {epoch + 1}/{args.num_train_epochs}, loss = {epoch_loss / num_items:.4f}",
                    )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:

        final_model_path = os.path.join(save_path, "final_model")
        console.log(f"Saving final model to {final_model_path}...")

        if not os.path.exists(final_model_path):
            os.makedirs(final_model_path, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        best_model_path = os.path.join(save_path, "best_model")
        for file in os.listdir(best_model_path):
            if file.endswith(".pt"):
                best_model_path = os.path.join(best_model_path, file)
                break
        if os.path.exists(best_model_path):
            console.log(
                f"Loading best model from {best_model_path} for final evaluation"
            )
            state_dict = torch.load(best_model_path, map_location="cpu")
            missing_keys, unexpected_keys = unwrapped_model.load_state_dict(
                state_dict, strict=False
            )
            if len(missing_keys) > 0:
                console.log(f"Missing keys when loading best model: {missing_keys}")
            if len(unexpected_keys) > 0:
                console.log(
                    f"Unexpected keys when loading best model: {unexpected_keys}"
                )
            console.log("Best model loaded successfully")

        if unwrapped_model.config.use_lora == True:
            unwrapped_model.llm_model = unwrapped_model.llm_model.merge_and_unload()
            unwrapped_model.config.use_lora = False

        torch.save(
            unwrapped_model.state_dict(),
            os.path.join(final_model_path, "model_weight.pt"),
        )
        tokenizer.save_pretrained(final_model_path)
        console.log(f"Final model saved to {final_model_path}")

    accelerator.end_training()"""

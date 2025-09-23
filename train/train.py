import os
import gc
import torch
import wandb
import shutil
import transformers
import torch.distributed as dist
from functools import partial
from model.gnn import GRAPH_KEYS
from torch.utils.data import DataLoader
from data.core import Data
from data.loader import GLMFDataset, collate_fn
from model.model import GLMFModelForCausalLM
from accelerate import Accelerator
from transformers.trainer_utils import seed_worker
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from train.utils import (
    patch_model,
    save_checkpoint,
)
from train.test import validate
from utils.utils import log_ram_usage


# typing
from argparse import Namespace
from rich.console import Console
from transformers import PreTrainedTokenizer


def train(
    args: Namespace,
    dataset: GLMFDataset,
    console: Console,
    model: GLMFModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    continue_training: bool = False,
    start_step: int = -1,
    max_num_checkpoint: int = 5,
    mixed_precision: str = "bf16",
    collate_fn: callable = collate_fn,
):
    collate_fn_ = partial(
        collate_fn, tokenizer=dataset.llm_tokenizer, max_seq_length=args.max_seq_length
    )
    if args.num_gpu == 1:
        console.log("Training on single GPU with mode: train_single_gpu_accelerate")
        train_single_gpu_accelerate(
            args=args,
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
        console.log(
            f"Training on multi GPU - {args.num_gpu} GPUs - with mode: train_multi_gpu_accelerate"
        )
        train_multi_gpu_accelerate(
            args=args,
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


def train_single_gpu_accelerate(
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

    # init wandb
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb",
        project_dir=args.log_dir,
    )

    save_path = os.path.join(args.output_dir, args.name)
    console.log(f"Model will be saved to {save_path}...")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if os.path.exists(save_path):
        if continue_training == False:
            shutil.rmtree(save_path)
            os.makedirs(save_path, exist_ok=True)
    # Initialize W&B run if main process
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
        tr_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    va_loader = DataLoader(
        va_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    logging_train_data(
        console=console, datasets=(tr_dataset, va_dataset), tokenizer=tokenizer
    )
    device = accelerator.device
    config = model.config
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    global_step = 0
    previous_checkpoint_step = -1
    best_val_loss = 10000.0

    if continue_training == False:
        optimizer.zero_grad()

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

            for step, batch in enumerate(tr_loader):

                if (continue_training == True) and (global_step <= start_step):
                    global_step += args.batch_size
                    ram_usage = log_ram_usage()
                    progress.update(
                        train_epoch_task,
                        advance=1,
                        description=f"Batch {step + 1}/{len(tr_loader)}: loss = N/A - RAM usage: {ram_usage:.1f} MB",
                    )
                    continue

                global_step += 1
                batch_loss = 0.0
                batch_size = batch["input"]["input_ids"].size(0)

                # batch_input = batch["input"].copy()
                if "token_type_ids" in batch["input"]:
                    batch["input"].pop("token_type_ids")

                micro_input = {
                    "input_ids": batch["input"]["input_ids"].to(device),
                    "attention_mask": batch["input"]["attention_mask"].to(device),
                    "labels": batch["input"]["labels"].to(device),
                }

                if "graph" in args.baseline_prompt:
                    graphs = []
                    graph_masks = []
                    graph_token_indices = []

                    for i in range(batch_size):
                        graph = batch["graph"][i]
                        for key in GRAPH_KEYS:
                            if key in graph.keys():
                                graph[key] = graph[key].to(device)

                        graph_mask = batch["graph_mask"][i].to(device)
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == model.config.graph_token_id[1]
                        )[1].tolist()
                        graphs.append(graph)
                        graph_masks.append(graph_mask)
                        graph_token_indices.append(graph_token_index)
                else:
                    graphs = None
                    graph_masks = None
                    graph_token_indices = None

                with accelerator.accumulate(model):
                    outputs = model(
                        **micro_input,
                        step=global_step,
                        graphs=graphs,
                        graph_masks=graph_masks,
                        graph_token_indices=graph_token_indices,
                    )

                    loss = outputs.loss
                    accelerator.backward(loss)

                    # compute the gradient norm of the nvib layer
                    if args.fuzz_model:
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                if "nvib_layer" in name and param.requires_grad:
                                    if param.grad is not None:
                                        grad_norm = param.grad.norm(2).item()
                                        console.log(
                                            f"Step {global_step}: Gradient norm of {name}: {grad_norm:.4f}"
                                        )

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                batch_loss += loss.item()

                avg_batch_loss = batch_loss / batch_size
                ram_usage = log_ram_usage()
                progress.update(
                    train_epoch_task,
                    advance=1,
                    description=f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f} - RAM usage: {ram_usage:.1f} MB",
                )
                epoch_loss += avg_batch_loss * batch_size
                num_items += batch_size

                for key in micro_input.keys():
                    micro_input[key] = micro_input[key].to("cpu")
                if "graph" in args.baseline_prompt:
                    for key in GRAPH_KEYS:
                        if key in graph.keys():
                            graph[key] = graph[key].to("cpu")
                            graph.pop(key, None)
                    graph_mask = graph_mask.to("cpu")
                    del graph_mask, graph
                outputs.logits = outputs.logits.to("cpu")
                loss = loss.to("cpu")
                del outputs, loss, micro_input
                gc.collect()
                torch.cuda.empty_cache()

                if global_step % args.logging_steps == 0:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    accelerator.log(
                        {
                            "train/loss": avg_batch_loss,
                            "train/learning_rate": current_lr,
                            "train/step": global_step,
                        },
                        step=global_step,
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
                        model=model,
                        path=checkpoint_dir,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        global_step=global_step,
                        max_num_checkpoint=max_num_checkpoint,
                        seed=args.seed,
                    )
                    if accelerator.is_main_process:
                        accelerator.print(f"Saving checkpoint to {checkpoint_dir}")

                if global_step % args.validating_steps == 0:
                    val_loss = validate(
                        args=args,
                        loader=va_loader,
                        model=model,
                        device=device,
                        accelerator=accelerator,
                        console=console,
                        config=config,
                        progress=progress,
                    )
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
                            os.path.join(checkpoint_dir, "model_weight.pt"),
                        )
                        tokenizer.save_pretrained(checkpoint_dir)

                        if accelerator.is_main_process:
                            accelerator.print(
                                f"Saving best checkpoint to {checkpoint_dir}"
                            )

                        del unwrapped_model
                        del checkpoint_dir
                        gc.collect()

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
    best_model_path = os.path.join(save_path, "best_model", "model_weight.pt")
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
    local_rank = model.rank
    # console.log(f"Local rank: {local_rank} of the training process")

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
        dataloader_params["worker_init_fn"] = seed_worker

    tr_loader = DataLoader(tr_dataset, **dataloader_params)
    va_loader = DataLoader(
        va_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    console.log(
        f"[green]Forward function before patching model: {transformers.modeling_flash_attention_utils._flash_attention_forward}[/green]\n\n"
    )
    patch_model(process_group=process_group)
    console.log(
        f"[cyan]Forward function after patching model: {transformers.modeling_flash_attention_utils._flash_attention_forward}[/cyan]\n\n"
    )
    console.log("Model patched with ring attention")
    device = accelerator.device
    config = model.config
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    console.log(f"Model prepared with accelerator: {model}")

    if accelerator.is_main_process:
        accelerator.print(f"***** Running training *****")
        accelerator.print(f"  Num examples = {len(tr_dataset)}")
        accelerator.print(f"  Instantaneous batch size per device = {args.batch_size}")
        accelerator.print(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        accelerator.print(f"  Total train batch size = {args.batch_size}")
        accelerator.print(
            f"  Total optimization steps = {args.num_train_epochs * len(tr_loader)}"
        )

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
                # batch_input = batch["input"].copy()
                if "token_type_ids" in batch["input"]:
                    batch["input"].pop("token_type_ids")

                micro_input = {
                    "input_ids": batch["input"]["input_ids"].to(device),
                    "attention_mask": batch["input"]["attention_mask"].to(device),
                    "labels": batch["input"]["labels"].to(device),
                }

                accelerator.wait_for_everyone()

                if "graph" in args.baseline_prompt:
                    graphs = []
                    graph_masks = []
                    graph_token_indices = []

                    for i in range(batch_size):
                        graph = batch["graph"][i]
                        for key in GRAPH_KEYS:
                            if key in graph.keys():
                                graph[key] = graph[key].to(device)

                        graph_mask = batch["graph_mask"][i].to(device)
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == model.config.graph_token_id[1]
                        )[1].tolist()
                        graphs.append(graph)
                        graph_masks.append(graph_mask)
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
                    )
                    accelerator.wait_for_everyone()
                    loss = outputs.loss

                    accelerator.backward(loss)
                    accelerator.wait_for_everyone()

                    if args.fuzz_model:
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                if "nvib_layer" in name and param.requires_grad:
                                    if param.grad is not None:
                                        grad_norm = param.grad.norm(2).item()
                                        console.log(
                                            f"Step {global_step} - rank {local_rank}: Gradient norm of {name}: {grad_norm:.4f}"
                                        )

                    if accelerator.sync_gradients:
                        accelerator.wait_for_everyone()
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        model.zero_grad()

                    with torch.no_grad():
                        all_losses = accelerator.gather(loss)
                        # fill all_losses with zero if it is NaN with torch.where
                        all_losses = torch.where(
                            torch.isnan(all_losses),
                            torch.zeros_like(all_losses),
                            all_losses,
                        )
                        console.log(
                            f"Step {global_step}: gathered losses: {all_losses}"
                        )
                        total_loss = torch.sum(all_losses)
                        batch_loss += total_loss.detach().float().item()

                for key in micro_input.keys():
                    micro_input[key] = micro_input[key].to("cpu")
                if "graph" in args.baseline_prompt:
                    for key in GRAPH_KEYS:
                        if key in graph.keys():
                            graph[key] = graph[key].to("cpu")
                            graph.pop(key, None)
                    graph_mask = graph_mask.to("cpu")
                    del graph_mask, graph
                outputs.logits = outputs.logits.to("cpu")
                loss = loss.to("cpu")
                del outputs, loss, micro_input
                gc.collect()
                torch.cuda.empty_cache()

                avg_batch_loss = batch_loss / batch_size
                if accelerator.is_main_process:
                    ram_usage = log_ram_usage()
                    console.log(
                        f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f} - RAM usage: {ram_usage:.1f} MB"
                    )
                    progress.update(
                        train_epoch_task,
                        advance=1,
                        description=f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f} - RAM usage: {ram_usage:.1f} MB",
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

                if accelerator.sync_gradients and (global_step % args.save_steps == 0):
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
                            print(f"Removing oldest checkpoint: {oldest_checkpoint}")
                            shutil.rmtree(oldest_checkpoint)

                        unwrapped_model = accelerator.unwrap_model(model)
                        save_checkpoint(
                            model=unwrapped_model,
                            path=checkpoint_dir_new,
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            global_step=global_step,
                            max_num_checkpoint=max_num_checkpoint,
                            seed=args.seed,
                        )
                        accelerator.print(f"Saving checkpoint to {checkpoint_dir_new}")
                        del unwrapped_model

                if global_step % args.validating_steps == 0:
                    accelerator.wait_for_everyone()
                    val_loss = validate(
                        args=args,
                        loader=va_loader,
                        model=model,
                        device=device,
                        config=config,
                        console=console,
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
                            # if unwrapped_model.config.use_lora == True:
                            #     unwrapped_model.llm_model = (
                            #         unwrapped_model.llm_model.merge_and_unload()
                            #     )
                            #     unwrapped_model.config.use_lora = False
                            torch.save(
                                unwrapped_model.state_dict(),
                                os.path.join(checkpoint_dir, "model_weight.pt"),
                            )
                            tokenizer.save_pretrained(checkpoint_dir)
                            accelerator.print(
                                f"Saving best checkpoint to {checkpoint_dir}"
                            )
                            del unwrapped_model
                            del checkpoint_dir
                            gc.collect()
                    model.train()

                if args.debug:
                    # only run 1 step in debug mode
                    break

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

            if args.debug:
                # only run 1 step in debug mode
                break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:

        final_model_path = os.path.join(save_path, "final_model")
        console.log(f"Saving final model to {final_model_path}...")

        if not os.path.exists(final_model_path):
            os.makedirs(final_model_path, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        best_model_path = os.path.join(save_path, "best_model", "model_weight.pt")
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

    accelerator.end_training()

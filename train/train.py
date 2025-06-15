import os
import gc
import torch
import wandb
import shutil
import traceback
from model.gnn import GRAPH_KEYS
from torch.utils.data import DataLoader
from data.core import Data
from data.loader import GLMFDataset, collate_fn
from model.model import GLMFModelForCausalLM, GLMFModelConfig
from accelerate import Accelerator
from transformers.trainer_utils import seed_worker
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from train.utils import patch_model, move_model_to_device, save_checkpoint
from train.test import validate
from utils.utils import log_ram_usage


# typing
from argparse import Namespace
from rich.console import Console


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
            collate_fn=collate_fn,
            # rank=-1,
        )
    else:
        console.log("Training on multi GPU with mode: train_multi_gpu_accelerate")
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
            collate_fn=collate_fn,
            # rank=-1,
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

    device = accelerator.device
    accelerator.print(f"Using {accelerator.num_processes} devices")
    accelerator.print(f"Mixed precision: {mixed_precision}")

    if args.model_debug == False:
        tr_dataset = GLMFDataset(
            data=dataset.train_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
            n_hops=dataset.n_hops,
        )
        va_dataset = GLMFDataset(
            data=dataset.val_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
            n_hops=dataset.n_hops,
        )
        tr_loader = DataLoader(
            tr_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
        )
        va_loader = DataLoader(
            va_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
        )
        console.log("Data prepared:")
        console.log(f"Train data: {len(tr_dataset)} data points")
        console.log(f"Valid data: {len(va_dataset)} data points")

    tokenizer = dataset.llm_tokenizer

    # Prepare everything with accelerator
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    global_step = 0
    previous_checkpoint_step = -1
    # Zero gradients initially.
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

            if ((continue_training == True) and (global_step > start_step)) or (
                continue_training == False
            ):
                train_epoch_task = progress.add_task(
                    f"Epoch {epoch + 1}/{args.num_train_epochs}", total=len(tr_loader)
                )

            epoch_loss = 0.0
            num_items = 0.0

            for step, batch in enumerate(tr_loader):

                if (continue_training == True) and (global_step <= start_step):
                    global_step += args.batch_size
                    continue

                global_step += args.batch_size
                batch_loss = 0.0
                batch_size = batch["input"]["input_ids"].size(0)

                # Process each sample in the batch as a micro-batch.
                for i in range(batch_size):

                    batch_input = batch["input"].copy()
                    if "token_type_ids" in batch_input:
                        batch_input.pop("token_type_ids")

                    micro_input = {
                        "input_ids": batch_input["input_ids"][i].to(device),
                        "attention_mask": batch_input["attention_mask"][i].to(device),
                        "labels": batch_input["labels"][i].to(device),
                    }

                    if args.debug:
                        log_info = f"Length of input_ids - {micro_input['input_ids'].size()}, attention_mask - {micro_input['attention_mask'].size()}, labels - {micro_input['labels'].size()}"
                        console.log(f"Step {global_step}: {log_info}")

                    if "graph" in args.baseline_prompt:
                        graph = batch["graph"][i]
                        for key in GRAPH_KEYS:
                            if key in graph.keys():
                                graph[key] = graph[key].to(device)

                        graph_mask = batch["graph_mask"][i].to(device)
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == model.config.graph_token_id[1]
                        )[1].tolist()
                    else:
                        graph = None
                        graph_mask = None
                        graph_token_index = None

                    with accelerator.accumulate(model):
                        outputs = model(
                            **micro_input,
                            graph=graph,
                            graph_mask=graph_mask,
                            graph_token_index=graph_token_index,
                        )

                        loss = outputs.loss
                        accelerator.backward(loss)

                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()

                    batch_loss += loss.item()

                avg_batch_loss = batch_loss / batch_size
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

                if (global_step % args.logging_steps == 0) and (args.debug == False):
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
                    accelerator.print(
                        f"Step {global_step}: Loss: {avg_batch_loss:.4f}, LR: {current_lr:.6f}"
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
                    # accelerator.save_state(checkpoint_dir)
                    if accelerator.is_main_process:
                        accelerator.print(f"Saving checkpoint to {checkpoint_dir}")

                del outputs, loss, micro_input, graph, graph_mask  # Free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if (global_step % args.validating_steps == 0) and (args.debug == False):
                    val_loss = validate(
                        args=args,
                        loader=va_loader,
                        model=model,
                        device=device,
                        accelerator=accelerator,
                        progress=progress,
                    )
                    wandb.log({"val_loss": val_loss})
                    console.log(
                        f"Validation loss: {val_loss:.4f} at step {global_step}"
                    )

                # if args.debug:
                #     break

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

            # if args.debug:
            #     break

    if model.config.use_lora == True:
        model.llm_model = model.llm_model.merge_and_unload()
        model.config.use_lora = False

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if any(p.device.type == "meta" for p in unwrapped_model.parameters()):
        device_to_save = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        move_model_to_device(unwrapped_model, device_to_save)

    final_model_path = os.path.join(save_path, "final_model")
    console.log(f"Saving final model to {final_model_path}...")

    if not os.path.exists(final_model_path):
        os.makedirs(final_model_path, exist_ok=True)
    # final_tokenizer_path = os.path.join(save_path, "tokenizer")

    # console.log(f"Config: {unwrapped_model.config}")

    # unwrapped_model.save_pretrained(
    #     final_model_path,
    #     is_main_process=accelerator.is_main_process,
    #     save_function=accelerator.save,
    # )

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

    local_rank = model.rank

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
    accelerator.print(f"Distributed type: {accelerator.distributed_type}")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Mixed precision: {mixed_precision}")

    if args.model_debug == False:
        tr_dataset = GLMFDataset(
            data=dataset.train_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
            n_hops=dataset.n_hops,
        )
        va_dataset = GLMFDataset(
            data=dataset.val_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
            n_hops=dataset.n_hops,
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
            va_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
        )
        console.log("Data prepared:")
        console.log(f"Train data: {len(tr_dataset)} data points")
        console.log(f"Valid data: {len(va_dataset)} data points")

    tokenizer = dataset.llm_tokenizer
    patch_model(model_type=model.config.model_type)
    console.log("Model patched with ring attention")
    device = accelerator.device
    config = model.config
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

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
    model.train()
    previous_checkpoint_step = -1

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

            if accelerator.is_main_process:
                # if ((continue_training == True) and (global_step > start_step)) or (
                #     continue_training == False
                # ):
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
                    console.log(
                        f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f} - RAM usage: {ram_usage:.1f} MB"
                    )
                    progress.update(
                        train_epoch_task,
                        advance=1,
                        description=f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f} - RAM usage: {ram_usage:.1f} MB",
                    )
                    continue

                global_step += args.batch_size
                batch_loss = 0.0
                batch_size = batch["input"]["input_ids"].size(0)

                # Process each sample in the batch as a micro-batch.
                for i in range(batch_size):

                    accelerator.wait_for_everyone()
                    batch_input = batch["input"].copy()
                    if "token_type_ids" in batch_input:
                        batch_input.pop("token_type_ids")

                    micro_input = {
                        "input_ids": batch_input["input_ids"][i].to(device),
                        "attention_mask": batch_input["attention_mask"][i].to(device),
                        "labels": batch_input["labels"][i].to(device),
                    }

                    accelerator.wait_for_everyone()
                    if "graph" in args.baseline_prompt:
                        if args.debug:
                            console.log("=" * 10 + "\n")
                            console.log(
                                f"Step {global_step}: Graph found in batch, checking keys..."
                            )
                            check_graph_exist_dict = {}
                            for key in GRAPH_KEYS:
                                check_graph_exist_dict[key] = False
                        graph = batch["graph"][i]
                        for key in GRAPH_KEYS:
                            if key in graph.keys():
                                graph[key] = graph[key].to(device)
                                if args.debug:
                                    check_graph_exist_dict[key] = True
                        if args.debug:
                            for key in GRAPH_KEYS:
                                if check_graph_exist_dict[key] == False:
                                    console.log(f"=" * 10 + "\n")
                                    console.log(
                                        f"[red]Step {global_step}: Graph {key} not found in batch !!!!"
                                    )
                            console.log("Done checking graph keys")
                            console.log(f"=" * 10 + "\n")
                        graph_mask = batch["graph_mask"][i].to(device)
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == config.graph_token_id[1]
                        )[1].tolist()
                    else:
                        graph = None
                        graph_mask = None
                        graph_token_index = None

                    accelerator.wait_for_everyone()
                    if args.debug & accelerator.is_main_process:
                        console.log(f"Step {global_step}: processed data")

                    with accelerator.accumulate(model):
                        try:
                            outputs = model(
                                **micro_input,
                                graph=graph,
                                graph_mask=graph_mask,
                                graph_token_index=graph_token_index,
                                step=global_step,
                                accelerator=accelerator,
                            )
                        except Exception as e:
                            console.log(
                                "Uncaught exception:\n" + traceback.format_exc()
                            )
                            raise
                        accelerator.wait_for_everyone()

                        if args.debug & accelerator.is_main_process:
                            console.log(
                                "=" * 10
                                + "\n"
                                + f"Step {global_step}: completed forward pass"
                                + "\n"
                                + "=" * 10
                            )

                        loss = outputs.loss
                        accelerator.backward(loss)
                        if args.debug & accelerator.is_main_process:
                            console.log(f"Step {global_step}: completed backward pass")

                        if accelerator.sync_gradients:
                            accelerator.wait_for_everyone()
                            if args.debug & accelerator.is_main_process:
                                console.log(
                                    f"Step {global_step}: Everyone waited for gradients"
                                )
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)
                            if args.debug & accelerator.is_main_process:
                                console.log(f"Step {global_step}: Clipped gradients")

                                console.log("=" * 100)
                                console.log(
                                    f"Step {global_step} - rank {local_rank}: Checking unused parameters"
                                )
                                found = False
                                for name, param in model.named_parameters():
                                    if param.grad is None and param.requires_grad:
                                        if found == False:
                                            console.log(
                                                f"Step {global_step} - rank {local_rank}: Found unused parameters:"
                                            )
                                            found = True
                                        console.log(f"{name}")
                                if found == False:
                                    console.log(
                                        f"Step {global_step} - rank {local_rank}: No unused parameters found"
                                    )
                                console.log("=" * 100)
                                console.log("\n" * 5)

                            optimizer.step()
                            if args.debug & accelerator.is_main_process:
                                console.log(
                                    f"Step {global_step}: Updated model parameters"
                                )
                            lr_scheduler.step()
                            if args.debug & accelerator.is_main_process:
                                console.log(
                                    f"Step {global_step}: Updated scheduler - learning rate is: {lr_scheduler.get_last_lr()[0]}"
                                )
                            optimizer.zero_grad()
                            if args.debug & accelerator.is_main_process:
                                console.log(f"Step {global_step}: Updated scheduler")

                    batch_loss += loss.detach().float()

                    for key in micro_input.keys():
                        micro_input[key] = micro_input[key].to("cpu")
                    if "graph" in args.baseline_prompt:
                        for key in GRAPH_KEYS:
                            if key in graph.keys():
                                graph[key] = graph[key].to("cpu")
                                graph.pop(key, None)
                        graph_mask = graph_mask.to("cpu")
                        del graph_mask, graph
                    del outputs, loss, micro_input
                    gc.collect()
                    torch.cuda.empty_cache()

                # if args.debug:
                #     pass

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
                    (global_step % args.logging_steps == 0)
                    and accelerator.is_main_process
                    and (args.debug == False)
                ):
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
                    accelerator.print(
                        f"Step {global_step}: Loss: {avg_batch_loss:.4f}, LR: {current_lr:.6f}"
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

                # del outputs, loss, micro_input, graph, graph_mask  # Free memory
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()

                if global_step % args.validating_steps == 0:
                    accelerator.wait_for_everyone()
                    val_loss = validate(
                        args=args,
                        loader=va_loader,
                        model=model,
                        device=device,
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
                    model.train()

            # if args.debug:
            #     break

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

        if unwrapped_model.config.use_lora == True:
            unwrapped_model.llm_model = unwrapped_model.llm_model.merge_and_unload()
            unwrapped_model.config.use_lora = False

        # if any(p.device.type == "meta" for p in unwrapped_model.parameters()):
        #     device_to_save = (
        #         torch.device("cuda")
        #         if torch.cuda.is_available()
        #         else torch.device("cpu")
        #     )
        #     move_model_to_device(unwrapped_model, device_to_save)

        # for p in unwrapped_model.parameters():
        #     console.log(f"Parameter device: {p.device}")
        #     if p.device.type == "meta":
        #         console.log("Model has meta parameters. Converting to CPU.")
        # break

        # unwrapped_model.save_pretrained(
        #     final_model_path,
        #     is_main_process=accelerator.is_main_process,
        #     save_function=accelerator.save,
        # )
        torch.save(
            unwrapped_model.state_dict(),
            os.path.join(final_model_path, "model_weight.pt"),
        )
        tokenizer.save_pretrained(final_model_path)
        console.log(f"Final model saved to {final_model_path}")

        # Log final model to W&B
        # if wandb.run is not None:
        #     os.makedirs(os.path.join(save_path, "antifact"), exist_ok=True)
        #     model_artifact = wandb.Artifact(
        #         name=f"model-{wandb.run.id}",
        #         type="model",
        #         description=f"Final model checkpoint",
        #     )
        #     model_artifact.add_dir(os.path.join(save_path, "antifact"))
        #     wandb.log_artifact(model_artifact)

    # End W&B run
    accelerator.end_training()

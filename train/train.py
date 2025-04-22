import os
import torch
import wandb
from torch.utils.data import DataLoader
from utils.constant import (
    GRAPH_START_TOKEN,
    GRAPH_PAD_TOKEN,
    GRAPH_END_TOKEN,
)
import torch.distributed as dist
from torch.distributed import barrier
from data.loader import GLMFDataset, collate_fn
from model.model import GLMFModelForCausalLM, GLMFModelConfig
from transformers import get_scheduler
from accelerate import Accelerator
from torch.optim import AdamW
from transformers.trainer_utils import seed_worker
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from train.utils import patch_model
from train.test import validate


# typing
from argparse import Namespace
from rich.console import Console


def train(
    args: Namespace,
    dataset: GLMFDataset,
    console: Console,
    device: torch.device,
    model: GLMFModelForCausalLM,
    collate_fn: callable = collate_fn,
    rank: int = 0,
):
    if args.use_accelerate:
        if args.num_gpu == 1:
            console.log("Training on single GPU with mode: train_single_gpu_accelerate")
            train_single_gpu_accelerate(
                args=args,
                dataset=dataset,
                console=console,
                model=model,
                collate_fn=collate_fn,
                rank=-1,
            )
        else:
            console.log("Training on multi GPU with mode: train_multi_gpu_accelerate")
            train_multi_gpu_accelerate(
                args=args,
                dataset=dataset,
                console=console,
                model=model,
                collate_fn=collate_fn,
                rank=-1,
            )
    else:
        if args.num_gpu == 1:
            console.log("Training on single GPU with mode: train_single_gpu")
            train_single_gpu(
                args=args,
                dataset=dataset,
                console=console,
                device=device,
                collate_fn=collate_fn,
            )
        elif args.num_gpu > 1:
            console.log("Training on multi GPU with mode: train_multi_gpu_ringattn")
            train_multi_gpu_ringattn(
                args=args,
                dataset=dataset,
                console=console,
                device=device,
                collate_fn=collate_fn,
                rank=rank,
            )


def train_single_gpu(
    args: Namespace,
    dataset: GLMFDataset,
    console: Console,
    device: torch.device,
    collate_fn: callable = collate_fn,
    rank: int = 0,
):

    # init wandb
    wandb.init(
        project="GLMFuzz",
        name=args.name,
        config=vars(args),
    )

    if args.model_debug == False:
        tr_dataset = GLMFDataset(
            data=dataset.train_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
        )
        va_dataset = GLMFDataset(
            data=dataset.val_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
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
    config = GLMFModelConfig(
        llm_model=args.llm_model,
        use_lora=args.use_lora,
        dtype=args.dtype,
        mode=args.gnn_mode,
        in_feats=args.in_feats,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        num_head=args.num_head,
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    if config.model_type not in ["llama", "qwen2"]:
        raise ValueError(
            f"Model type {config.model_type} is not supported. Please use 'llama' or 'qwen2'."
        )

    # if args.debug:
    #     console.log(f"Model config initialized: {config}")

    model = GLMFModelForCausalLM(
        config=config,
        tokenizer=tokenizer,
        baseline_prompt=args.baseline_prompt,
        debug=args.debug,
        rank=rank,
    )

    # if args.debug:
    #     console.log(f"Model initialized with config: {config}")

    model.config.graph_token_id = [
        tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
        tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
        tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
    ]

    console.log(
        f"Special tokens added to tokenizer and model: {model.config.graph_token_id}"
    )
    if args.model_debug:
        return

    if args.debug:
        console.log("Model & tokenizer loaded")

    model.to(device)
    # model.gnn.to("cpu")
    model.train()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )
    global_step = 0
    loss_track = []

    # Zero gradients initially.
    optimizer.zero_grad()

    with Progress(
        SpinnerColumn(),  # Shows a spinner
        TextColumn(
            "[progress.description]{task.description}"
        ),  # Displays additional info
        BarColumn(),  # Displays a progress bar
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # Shows percentage
    ) as progress:

        train_task = progress.add_task("Training...", total=args.num_train_epochs)

        for epoch in range(args.num_train_epochs):
            model.train()

            train_epoch_task = progress.add_task(
                f"Epoch {epoch + 1}/{args.num_train_epochs}", total=len(tr_loader)
            )

            epoch_loss = 0.0
            num_items = 0.0

            for step, batch in enumerate(tr_loader):
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

                    # if args.debug:
                    #     console.log(f"Micro input: {micro_input}")

                    graph = batch["graph"][i]
                    for key in model.gnn.type_of_graph:
                        if key in graph.keys():
                            graph[key] = graph[key].to(device)

                    graph_mask = batch["graph_mask"][i].to(device)

                    if "graph" in args.baseline_prompt:
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == model.config.graph_token_id[1]
                        )[1].tolist()
                        # if args.debug:
                        #     console.log(f"Graph token id: {graph_token_index}")
                    else:
                        graph_token_index = None

                    outputs = model(
                        **micro_input,
                        graph=graph,
                        graph_mask=graph_mask,
                        graph_token_index=graph_token_index,
                    )

                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                    batch_loss += (
                        outputs.loss.item()
                    )  # For logging (using the unscaled loss).

                    # Update parameters once enough gradients have been accumulated.
                    if global_step % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=args.max_grad_norm
                        )
                        optimizer.step()  # Update parameters.
                        optimizer.zero_grad()  # Reset gradients.

                # Log average loss for this batch.
                avg_batch_loss = batch_loss / batch_size
                progress.update(
                    train_epoch_task,
                    advance=1,
                    description=f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f}",
                )
                epoch_loss += avg_batch_loss * batch_size
                num_items += batch_size

                if global_step % args.logging_steps == 0:
                    wandb.log({"train_loss": avg_batch_loss})

                if global_step % args.validating_steps == 0:
                    val_loss = validate(
                        args=args, loader=va_loader, model=model, device=device
                    )
                    wandb.log({"val_loss": val_loss})
                    console.log(
                        f"Validation loss: {val_loss:.4f} at step {global_step}"
                    )
                    # model.train()

            progress.remove_task(train_epoch_task)
            progress.update(
                train_task,
                advance=1,
                description=f"Epoch {epoch + 1}/{args.num_train_epochs}, loss = {epoch_loss / num_items:.4f}",
            )

    if model.config.use_lora == True:
        model.llm_model = model.llm_model.merge_and_unload()
        model.config.use_lora = False

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def train_multi_gpu_ringattn(
    args: Namespace,
    dataset: GLMFDataset,
    console: Console,
    device: torch.device = None,
    collate_fn: callable = collate_fn,
    rank: int = 0,
):

    if rank == 0:
        wandb.init(
            project="GLMFuzz",
            name=args.name,
            config=vars(args),
        )

    if args.model_debug == False:
        tr_dataset = GLMFDataset(
            data=dataset.train_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
        )
        va_dataset = GLMFDataset(
            data=dataset.val_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
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

        # if rank == 0:
        barrier()
        console.log("Data prepared:")
        console.log(f"Train data: {len(tr_dataset)} data points")
        console.log(f"Valid data: {len(va_dataset)} data points")

    tokenizer = dataset.llm_tokenizer
    config = GLMFModelConfig(
        llm_model=args.llm_model,
        use_lora=args.use_lora,
        dtype=args.dtype,
        mode=args.gnn_mode,
        in_feats=args.in_feats,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        num_head=args.num_head,
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    if config.model_type not in ["llama", "qwen2"]:
        raise ValueError(
            f"Model type {config.model_type} is not supported. Please use 'llama' or 'qwen2'."
        )

    # apply ring attention
    patch_model(model_type=config.model_type, mode="ring")
    console.log("Model patched with ring attention")

    if (args.debug) and (rank == 0):
        console.log(f"Model config initialized: {config}")

    model = GLMFModelForCausalLM(
        config=config,
        baseline_prompt=args.baseline_prompt,
        tokenizer=tokenizer,
        multi_gpu=True,
        debug=args.debug,
        rank=rank,
    )
    # tokenizer = dataset.llm_tokenizer
    # model.llm_model.resize_token_embeddings(len(tokenizer))
    if (args.debug) and (rank == 0):
        console.log(f"Model initialized with config: {config}")

    model.config.graph_token_id = [
        tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
        tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
        tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
    ]

    config.rope_theta = args.rope_theta
    config.max_position_embeddings = args.model_max_length

    model.to(device)
    model.train()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )
    global_step = 0
    loss_track = []

    # Zero gradients initially.
    optimizer.zero_grad()
    # model, optimizer, tr_loader = accelerator.prepare(model, optimizer, tr_loader)
    console.log(
        f"Model & optimizer prepared for multi-GPU training with device: {device}"
    )
    # run_nvidia_smi(console=console)
    barrier()

    with Progress(
        SpinnerColumn(),  # Shows a spinner
        TextColumn(
            "[progress.description]{task.description}"
        ),  # Displays additional info
        BarColumn(),  # Displays a progress bar
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # Shows percentage
    ) as progress:

        if rank == 0:
            train_task = progress.add_task("Training...", total=args.num_train_epochs)

        for epoch in range(args.num_train_epochs):
            model.train()

            if rank == 0:
                train_epoch_task = progress.add_task(
                    f"Epoch {epoch + 1}/{args.num_train_epochs}", total=len(tr_loader)
                )

            epoch_loss = 0.0
            num_items = 0.0

            for step, batch in enumerate(tr_loader):
                global_step += 1
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

                    # if args.debug:
                    #     console.log("Logging nvidia-smi with micro_input")
                    #     # run_nvidia_smi(console=console)
                    #     console.log(
                    #         "=" * 100
                    #         + "\n" * 2
                    #         + f"Micro input at rank {rank} | step {global_step}: {[micro_input[key].size() for key in micro_input]}"
                    #         + "\n" * 2
                    #         + "=" * 100
                    #     )

                    graph = batch["graph"][i]
                    for key in model.gnn.type_of_graph:
                        if key in graph.keys():
                            graph[key] = graph[key].to(device)

                    graph_mask = batch["graph_mask"][i].to(device)

                    if "graph" in args.baseline_prompt:
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == model.config.graph_token_id[1]
                        )[1].tolist()
                        # if (args.debug) and (rank == 0):
                        #     console.log(f"Graph token id: {graph_token_index}")
                    else:
                        graph_token_index = None

                    outputs = model(
                        **micro_input,
                        graph=graph,
                        graph_mask=graph_mask,
                        graph_token_index=graph_token_index,
                        step=global_step,
                    )

                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
                    loss.backward()

                    batch_loss += outputs.loss.item()
                    del outputs, loss, micro_input, graph, graph_mask  # Free memory
                    torch.cuda.empty_cache()
                    barrier()
                    # For logging (using the unscaled loss).

                    # Update parameters once enough gradients have been accumulated.
                    if global_step % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=args.max_grad_norm
                        )
                        optimizer.step()  # Update parameters.
                        optimizer.zero_grad(set_to_none=True)  # Reset gradients.

                # Log average loss for this batch.
                avg_batch_loss = batch_loss / batch_size
                if rank == 0:
                    progress.update(
                        train_epoch_task,
                        advance=1,
                        description=f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f}",
                    )
                epoch_loss += avg_batch_loss * batch_size
                num_items += batch_size

                if global_step % args.logging_steps == 0:
                    if rank == 0:
                        wandb.log({"train_loss": avg_batch_loss})

                if global_step % args.validating_steps == 0:
                    if rank == 0:
                        val_loss = validate(
                            args=args, loader=va_loader, model=model, device=device
                        )
                        wandb.log({"val_loss": val_loss})
                        console.log(
                            f"Validation loss: {val_loss:.4f} at step {global_step}"
                        )

            if rank == 0:
                progress.remove_task(train_epoch_task)
                progress.update(
                    train_task,
                    advance=1,
                    description=f"Epoch {epoch + 1}/{args.num_train_epochs}, loss = {epoch_loss / num_items:.4f}",
                )

    if rank == 0:
        if model.config.use_lora == True:
            model.llm_model = model.llm_model.merge_and_unload()
            model.config.use_lora = False

        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


def train_single_gpu_accelerate(
    args: Namespace,
    dataset: GLMFDataset,
    console: Console,
    collate_fn: callable = collate_fn,
    model: GLMFModelForCausalLM = None,
    rank: int = 0,
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
        )
        va_dataset = GLMFDataset(
            data=dataset.val_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
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
    # config = GLMFModelConfig(
    #     llm_model=args.llm_model,
    #     use_lora=args.use_lora,
    #     dtype=args.dtype,
    #     mode=args.gnn_mode,
    #     in_feats=args.in_feats,
    #     n_hidden=args.n_hidden,
    #     n_layers=args.n_layers,
    #     num_head=args.num_head,
    #     dropout=args.dropout,
    #     lora_r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     lora_dropout=args.lora_dropout,
    #     lora_target_modules=args.lora_target_modules,
    #     device_map="cuda" if torch.cuda.is_available() else "cpu",
    # )

    # if config.model_type not in ["llama", "qwen2"]:
    #     raise ValueError(
    #         f"Model type {config.model_type} is not supported. Please use 'llama' or 'qwen2'."
    #     )

    # model = GLMFModelForCausalLM(
    #     config=config,
    #     tokenizer=tokenizer,
    #     baseline_prompt=args.baseline_prompt,
    #     debug=args.debug,
    #     rank=rank,
    # )
    model.llm_model.gradient_checkpointing_enable()

    model.config.graph_token_id = [
        tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
        tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
        tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
    ]
    if args.model_debug:
        return

    if args.debug:
        console.log("Model & tokenizer loaded")
        console.log(
            f"Special tokens added to tokenizer and model: {model.config.graph_token_id}"
        )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=args.num_train_epochs,
    )

    # Prepare everything with accelerator
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    global_step = 0

    # Zero gradients initially.
    optimizer.zero_grad()

    with Progress(
        SpinnerColumn(),  # Shows a spinner
        TextColumn(
            "[progress.description]{task.description}"
        ),  # Displays additional info
        BarColumn(),  # Displays a progress bar
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # Shows percentage
    ) as progress:

        train_task = progress.add_task("Training...", total=args.num_train_epochs)

        for epoch in range(args.num_train_epochs):
            model.train()

            train_epoch_task = progress.add_task(
                f"Epoch {epoch + 1}/{args.num_train_epochs}", total=len(tr_loader)
            )

            epoch_loss = 0.0
            num_items = 0.0

            for step, batch in enumerate(tr_loader):
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

                    # if args.debug:
                    #     console.log(f"Micro input: {micro_input}")

                    graph = batch["graph"][i]
                    for key in model.gnn.type_of_graph:
                        if key in graph.keys():
                            graph[key] = graph[key].to(device)

                    graph_mask = batch["graph_mask"][i].to(device)

                    if "graph" in args.baseline_prompt:
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == model.config.graph_token_id[1]
                        )[1].tolist()
                    else:
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
                progress.update(
                    train_epoch_task,
                    advance=1,
                    description=f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f}",
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
                    checkpoint_dir = os.path.join(
                        save_path,
                        f"checkpoint-{global_step}",
                    )
                    accelerator.save_state(checkpoint_dir)
                    if accelerator.is_main_process:
                        accelerator.print(f"Saving checkpoint to {checkpoint_dir}")

                del outputs, loss, micro_input, graph, graph_mask  # Free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if (global_step % args.validating_steps == 0) and (args.debug == False):
                    val_loss = validate(
                        args=args, loader=va_loader, model=model, device=device
                    )
                    wandb.log({"val_loss": val_loss})
                    console.log(
                        f"Validation loss: {val_loss:.4f} at step {global_step}"
                    )

                if args.debug:
                    break

            progress.remove_task(train_epoch_task)
            progress.update(
                train_task,
                advance=1,
                description=f"Epoch {epoch + 1}/{args.num_train_epochs}, loss = {epoch_loss / num_items:.4f}",
            )
            if args.debug:
                break

    if model.config.use_lora == True:
        model.llm_model = model.llm_model.merge_and_unload()
        model.config.use_lora = False

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    final_model_path = os.path.join(save_path, "final_model")
    console.log(f"Saving final model to {final_model_path}...")
    final_tokenizer_path = os.path.join(save_path, "tokenizer")

    console.log(f"Config: {unwrapped_model.config}")

    unwrapped_model.save_pretrained(
        final_model_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    tokenizer.save_pretrained(final_tokenizer_path)

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
    dataset: GLMFDataset,
    console: Console,
    collate_fn: callable = collate_fn,
    model: GLMFModelForCausalLM = None,
    rank: int = 0,
    mixed_precision: str = "bf16",
):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb",
        project_dir=args.log_dir,
    )
    if rank < 0:
        rank = accelerator.process_index

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
    accelerator.print(f"Distributed type: {accelerator.distributed_type}")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Mixed precision: {mixed_precision}")

    if args.model_debug == False:
        tr_dataset = GLMFDataset(
            data=dataset.train_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
        )
        va_dataset = GLMFDataset(
            data=dataset.val_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
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
    # config = GLMFModelConfig(
    #     llm_model=args.llm_model,
    #     use_lora=args.use_lora,
    #     dtype=args.dtype,
    #     mode=args.gnn_mode,
    #     in_feats=args.in_feats,
    #     n_hidden=args.n_hidden,
    #     n_layers=args.n_layers,
    #     num_head=args.num_head,
    #     dropout=args.dropout,
    #     lora_r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     lora_dropout=args.lora_dropout,
    #     lora_target_modules=args.lora_target_modules,
    #     device_map="cuda" if torch.cuda.is_available() else "cpu",
    # )

    # setattr(config, "use_cache", False)
    # if config.model_type not in ["llama", "qwen2"]:
    #     raise ValueError(
    #         f"Model type {config.model_type} is not supported. Please use 'llama' or 'qwen2'."
    #     )

    patch_model(model_type=model.config.model_type, mode="ring")
    console.log("Model patched with ring attention")

    model.gradient_checkpointing_enable()
    model.config.graph_token_id = [
        tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
        tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
        tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
    ]
    if args.model_debug:
        return

    # if args.debug:
    #     console.log("Model & tokenizer loaded")
    #     console.log(
    #         f"Special tokens added to tokenizer and model: {model.config.graph_token_id}"
    #     )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=args.num_train_epochs,
    )
    device = accelerator.device
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

    with Progress(
        SpinnerColumn(),  # Shows a spinner
        TextColumn(
            "[progress.description]{task.description}"
        ),  # Displays additional info
        BarColumn(),  # Displays a progress bar
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # Shows percentage
    ) as progress:

        if accelerator.is_main_process:
            train_task = progress.add_task("Training...", total=args.num_train_epochs)

        for epoch in range(args.num_train_epochs):
            model.train()

            if accelerator.is_main_process:
                train_epoch_task = progress.add_task(
                    f"Epoch {epoch + 1}/{args.num_train_epochs}", total=len(tr_loader)
                )

            epoch_loss = 0.0
            num_items = 0.0

            for step, batch in enumerate(tr_loader):
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

                    graph = batch["graph"][i]
                    for key in model.gnn.type_of_graph:
                        if key in graph.keys():
                            graph[key] = graph[key].to(device)

                    graph_mask = batch["graph_mask"][i].to(device)

                    if "graph" in args.baseline_prompt:
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == model.config.graph_token_id[1]
                        )[1].tolist()
                    else:
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
                            accelerator.wait_for_everyone()
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()

                    batch_loss += loss.detach().float()

                if args.debug:
                    break

                avg_batch_loss = batch_loss / batch_size
                if accelerator.is_main_process:
                    progress.update(
                        train_epoch_task,
                        advance=1,
                        description=f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f}",
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

                if (
                    accelerator.sync_gradients
                    and (global_step % args.save_steps == 0)
                    and accelerator.is_main_process
                ):
                    accelerator.wait_for_everyone()
                    checkpoint_dir = os.path.join(
                        save_path,
                        f"checkpoint-{global_step}",
                    )
                    accelerator.save_state(checkpoint_dir)
                    if accelerator.is_main_process:
                        accelerator.print(f"Saving checkpoint to {checkpoint_dir}")

                del outputs, loss, micro_input, graph, graph_mask  # Free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if (
                    (global_step % args.validating_steps == 0)
                    and accelerator.is_main_process
                    and (args.debug == False)
                ):

                    val_loss = validate(
                        args=args, loader=va_loader, model=model, device=device
                    )
                    wandb.log({"val_loss": val_loss})
                    console.log(
                        f"Validation loss: {val_loss:.4f} at step {global_step}"
                    )

            if args.debug:
                break

            if accelerator.is_main_process:
                progress.remove_task(train_epoch_task)
                progress.update(
                    train_task,
                    advance=1,
                    description=f"Epoch {epoch + 1}/{args.num_train_epochs}, loss = {epoch_loss / num_items:.4f}",
                )

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if unwrapped_model.config.use_lora == True:
        unwrapped_model.llm_model = unwrapped_model.llm_model.merge_and_unload()
        unwrapped_model.config.use_lora = False

    if accelerator.is_main_process:

        final_model_path = os.path.join(save_path, "final_model.pt")
        console.log(f"Saving final model to {final_model_path}...")
        final_tokenizer_path = os.path.join(save_path, "tokenizer")

        unwrapped_model.save_pretrained(
            final_model_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(final_tokenizer_path)

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

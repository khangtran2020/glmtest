import torch
import wandb
from torch.utils.data import DataLoader
from utils.constant import (
    GRAPH_START_TOKEN,
    GRAPH_PAD_TOKEN,
    GRAPH_END_TOKEN,
    FUZZ_START_TOKEN,
    FUZZ_END_TOKEN,
)
from data.loader import GLMFDataset, collate_fn
from model.model import GLMFModelForCausalLM, GLMFModelConfig
from train.utils import smart_tokenizer_and_embedding_resize
from transformers import AdamW, get_linear_schedule_with_warmup
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# typing
from argparse import Namespace
from rich.console import Console


def train(
    args: Namespace,
    dataset: GLMFDataset,
    console: Console,
    device: torch.device,
    collate_fn: callable = collate_fn,
):
    if args.num_gpu == 1:
        train_single_gpu(
            args=args,
            dataset=dataset,
            console=console,
            device=device,
            collate_fn=collate_fn,
        )
    elif args.num_gpu > 1:
        pass


def train_single_gpu(
    args: Namespace,
    dataset: GLMFDataset,
    console: Console,
    device: torch.device,
    collate_fn: callable = collate_fn,
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
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    if config.model_type not in ["llama", "qwen2"]:
        raise ValueError(
            f"Model type {config.model_type} is not supported. Please use 'llama' or 'qwen2'."
        )

    if args.debug:
        console.log(f"Model config initialized: {config}")

    model = GLMFModelForCausalLM(config=config, baseline_prompt=args.baseline_prompt)
    if args.debug:
        console.log(f"Model initialized with config: {config}")
    # special_tokens_dict = {
    #     "additional_special_tokens": [
    #         GRAPH_START_TOKEN,
    #         GRAPH_PAD_TOKEN,
    #         GRAPH_END_TOKEN,
    #         FUZZ_START_TOKEN,
    #         FUZZ_END_TOKEN,
    #     ]
    # }
    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=special_tokens_dict,
    #     tokenizer=tokenizer,
    #     model=model.llm_model,
    # )

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
    model.gnn.to("cpu")
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

                    if args.debug:
                        console.log(f"Micro input: {micro_input}")

                    graph = batch["graph"][i]
                    graph_mask = batch["graph_mask"][i]

                    outputs = model(
                        **micro_input,
                        graph=graph,
                        graph_mask=graph_mask,
                    )

                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                    batch_loss += (
                        outputs.loss.item()
                    )  # For logging (using the unscaled loss).

                    # Update parameters once enough gradients have been accumulated.
                    if global_step % args.gradient_accumulation_steps == 0:
                        optimizer.step()  # Update parameters.
                        optimizer.zero_grad()  # Reset gradients.

                # Log average loss for this batch.
                avg_batch_loss = batch_loss / batch_size
                train_epoch_task.update(
                    advance=1,
                    description=f"Batch {step + 1}/{len(tr_loader)}: loss = {avg_batch_loss:.4f}",
                )
                epoch_loss += avg_batch_loss * batch_size
                num_items += batch_size

                if step % args.logging_steps == 0:
                    wandb.log({"train_loss": avg_batch_loss})

                if step % args.validation_steps == 0:
                    model.eval()
                    with torch.no_grad():

                        val_loss = 0.0
                        num_item = 0
                        for step, batch in enumerate(va_loader):
                            batch_loss = 0.0
                            batch_size = batch["input"]["input_ids"].size(0)
                            num_item += batch_size

                            # Process each sample in the batch as a micro-batch.
                            for i in range(batch_size):
                                global_step += 1
                                batch_input = batch["input"].copy()
                                if "token_type_ids" in batch_input:
                                    batch_input.pop("token_type_ids")
                                micro_input = {
                                    "input_ids": batch_input["input_ids"][i].to(device),
                                    "attention_mask": batch_input["attention_mask"][
                                        i
                                    ].to(device),
                                    "labels": batch_input["labels"][i].to(device),
                                }

                                graph = batch["graph"][i]
                                graph_mask = batch["graph_mask"][i]
                                outputs = model(
                                    **micro_input,
                                    graph=graph,
                                    graph_mask=graph_mask,
                                )
                                loss = outputs.loss
                                batch_loss += loss.item()

                            val_loss += batch_loss
                        val_loss /= num_item
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

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

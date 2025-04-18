import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.loader import GLMFDataset, collate_fn
from model.model import GLMFModelForCausalLM
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn


# typing
from argparse import Namespace
from rich.console import Console


def test(
    args: Namespace,
    data: GLMFDataset,
    model: GLMFModelForCausalLM,
    console: Console,
    collate_fn: callable = collate_fn,
):
    data.testing = True
    loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    console.log(f"Test data: {len(loader)} data points")
    console.log("Testing...")

    with Progress(
        SpinnerColumn(),  # Shows a spinner
        TextColumn(
            "[progress.description]{task.description}"
        ),  # Displays additional info
        BarColumn(),  # Displays a progress bar
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # Shows percentage
    ) as progress:
        test_task = progress.add_task("Testing...", total=len(loader))
        with torch.no_grad():
            generated_text = []
            for step, batch_data in enumerate(loader):

                uuid, batch = batch_data
                batch_size = batch["input"]["input_ids"].size(0)

                # Process each sample in the batch as a micro-batch.
                for i in range(batch_size):
                    batch_input = batch["input"].copy()
                    if "token_type_ids" in batch_input:
                        batch_input.pop("token_type_ids")
                    micro_input = {
                        "input_ids": batch_input["input_ids"][i].to(model.device),
                        "attention_mask": batch_input["attention_mask"][i].to(
                            model.device
                        ),
                        "labels": batch_input["labels"][i].to(model.device),
                    }

                    graph = batch["graph"][i]
                    for key in model.gnn.type_of_graph:
                        if key in graph.keys():
                            graph[key] = graph[key].to(model.device)

                    graph_mask = batch["graph_mask"][i].to(model.device)

                    if "graph" in args.baseline_prompt:
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == model.config.graph_token_id[1]
                        )[1].tolist()
                        if args.debug:
                            console.log(f"Graph token id: {graph_token_index}")

                        outputs = model.generate(
                            inputs=micro_input["input_ids"],
                            graph=graph,
                            graph_mask=graph_mask,
                            graph_token_index=graph_token_index,
                            max_new_tokens=args.max_new_tokens,
                        )
                    else:
                        graph_token_index = None
                        outputs = model.generate(
                            inputs=micro_input["input_ids"],
                            graph_token_index=graph_token_index,
                            max_new_tokens=args.max_new_tokens,
                        )

                    generated_text.append({uuid: outputs})

                progress.update(
                    test_task,
                    advance=1,
                    description=f"Testing... {step}/{len(loader)}",
                )
    console.log("Testing finished.")
    save_dir = os.path.join(args.gen_dir, f"{args.name}.jsonl")
    with console.status("Saving results..."):
        # save generated text to jsonl file
        with open(save_dir, "w") as f:
            for item in generated_text:
                for key, value in item.items():
                    f.write(f"{key}: {value}\n")
    console.log(f"Results saved to {save_dir}")


def validate(args, loader, model, device):
    model.eval()
    with torch.no_grad():

        val_loss = 0.0
        num_item = 0

        with tqdm(
            total=len(loader),
            position=0,
            leave=True,
            ncols=80,
            dynamic_ncols=True,
            mininterval=1.0,
            smoothing=0.1,
        ) as pbar:

            for step, batch in enumerate(loader):
                batch_loss = 0.0
                batch_size = batch["input"]["input_ids"].size(0)
                num_item += batch_size

                # Process each sample in the batch as a micro-batch.
                try:
                    for i in range(batch_size):
                        batch_input = batch["input"].copy()
                        if "token_type_ids" in batch_input:
                            batch_input.pop("token_type_ids")
                        micro_input = {
                            "input_ids": batch_input["input_ids"][i].to(device),
                            "attention_mask": batch_input["attention_mask"][i].to(
                                device
                            ),
                            "labels": batch_input["labels"][i].to(device),
                        }

                        graph = batch["graph"][i]
                        for key in model.gnn.type_of_graph:
                            if key in graph.keys():
                                graph[key] = graph[key].to(device)

                        graph_mask = batch["graph_mask"][i].to(device)

                        if "graph" in args.baseline_prompt:
                            graph_token_index = torch.where(
                                micro_input["input_ids"]
                                == model.config.graph_token_id[1]
                            )[1].tolist()
                        else:
                            graph_token_index = None

                        outputs = model(
                            **micro_input,
                            graph=graph,
                            graph_mask=graph_mask,
                            graph_token_index=graph_token_index,
                        )
                        loss = outputs.loss
                        batch_loss += loss.item()
                except torch.cuda.OutOfMemoryError as e:
                    tqdm.write(
                        f"OOM in batch {step}: input_dis {micro_input['input_ids'].size()} - graph_mask {graph_mask.size()}"
                    )
                    torch.cuda.empty_cache()
                    continue

                val_loss += batch_loss
                pbar.update(1)

        val_loss /= num_item
        return val_loss
        # model.train()

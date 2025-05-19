import os
import gc
import sys
import json
import time
import torch
from model.gnn import GRAPH_KEYS
from tqdm import tqdm
from data.loader import GLMFDataset, collate_fn
from model.model import GLMFModelForCausalLM, GLMFModelConfig

# from transformers import SinkCache
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from train.utils import patch_model, move_model_to_device, save_checkpoint
from torch.utils.data import DataLoader
from accelerate import Accelerator

# typing
from argparse import Namespace
from rich.console import Console


def test(
    args: Namespace,
    dataset: GLMFDataset,
    model: GLMFModelForCausalLM,
    console: Console,
    config: GLMFModelConfig = None,
    collate_fn: callable = collate_fn,
):
    if config is None:
        config = model.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    console.log("Testing on device ... :", device)
    te_dataset = GLMFDataset(
        data=dataset.test_data,
        tokenizer=dataset.llm_tokenizer,
        max_seq_length=args.max_seq_length,
        debug=args.debug,
        n_hops=dataset.n_hops,
        testing=True,
    )
    tokenizer = dataset.llm_tokenizer
    # past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
    # loader = DataLoader(te_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    console.log(f"Test data: {len(te_dataset)} data points")
    console.log("Testing...")

    with Progress(
        SpinnerColumn(),  # Shows a spinner
        TextColumn(
            "[progress.description]{task.description}"
        ),  # Displays additional info
        BarColumn(),  # Displays a progress bar
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # Shows percentage
    ) as progress:
        test_task = progress.add_task("Testing...", total=len(te_dataset))
        with torch.no_grad():
            generated_text = {}
            time_list = []
            for step, batch_data in enumerate(te_dataset):
                start_time = time.time()
                uuid, batch = batch_data
                # batch_size = batch["input"]["input_ids"].size(0)

                # Process each sample in the batch as a micro-batch.
                # for i in range(batch_size):
                batch_input = batch["input"].copy()
                if "token_type_ids" in batch_input:
                    batch_input.pop("token_type_ids")
                micro_input = {
                    "input_ids": batch_input["input_ids"].to(device),
                    "attention_mask": batch_input["attention_mask"].to(device),
                    "labels": None,
                }
                #if model.rank==0:
                if "graph" in args.baseline_prompt:
                    graph = batch["graph"]
                    for key in GRAPH_KEYS:
                        if key in graph.keys():
                            graph[key] = graph[key].to(device)

                    graph_mask = batch["graph_mask"].to(device)

                    graph_token_index = torch.where(
                        micro_input["input_ids"] == config.graph_token_id[1]
                    )[1].tolist()
                    # if args.debug:
                    #     console.log(f"Graph token id: {graph_token_index}")

                    outputs = model.generate(
                        inputs=micro_input["input_ids"],
                        graph=graph,
                        graph_mask=graph_mask,
                        graph_token_index=graph_token_index,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=False,
                    )
                else:
                    graph_token_index = None
                    outputs = model.generate(
                        inputs=micro_input["input_ids"],
                        graph_token_index=graph_token_index,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=False,
                    )
         
                out_text = tokenizer.batch_decode(outputs[:,micro_input["input_ids"].size(1):], skip_special_tokens=True)[0]
                if model.rank==0:
                    print(out_text)
                # exit()
                generated_text[uuid] = out_text
                end_time = time.time()
                process_time = end_time - start_time
                time_list.append(process_time)
                avg_time = sum(time_list) / len(time_list)
                progress.update(
                    test_task,
                    advance=1,
                    description=f"Testing... {step}/{len(te_dataset)} - {avg_time:.2f}s for 1 sample",
                )
    console.log("Testing finished.")
    save_dir = os.path.join(args.gen_dir, f"{args.name}.json")
    with console.status("Saving results..."):
        # save generated text to jsonl file
        with open(save_dir, "w", encoding="utf-8") as f:
            # save as json file
            json.dump(generated_text, f, ensure_ascii=False, indent=4)

    console.log(f"Results saved to {save_dir}")


def testCache(
    args: Namespace,
    dataset: GLMFDataset,
    model: GLMFModelForCausalLM,
    console: Console,
    collate_fn: callable = collate_fn,
):
    
    accelerator = Accelerator(
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        # mixed_precision=mixed_precision,
        # log_with="wandb",
        # project_dir=args.log_dir,
    )
    
    device = accelerator.device
    accelerator.print(f"Using {accelerator.num_processes} devices")
    # accelerator.print(f"Mixed precision: {mixed_precision}")
    
    
    
    console.log("Testing on device ... :", device)
    te_dataset = GLMFDataset(
        data=dataset.test_data,
        tokenizer=dataset.llm_tokenizer,
        max_seq_length=args.max_seq_length,
        debug=args.debug,
        n_hops=dataset.n_hops,
        testing=True,
    )
    tokenizer = dataset.llm_tokenizer

    ##Remove <|fuzz|> and <|/fuzz|> of the special token
    removed = ["<|fuzz|>", "<|/fuzz|>"]
    present = [t for t in removed if t in tokenizer.additional_special_tokens]
    # 1. filter out from the additional_special_tokens list
    new_ast = [
        t for t in tokenizer.additional_special_tokens
        if t not in present
    ]
    # update internal structures
    tokenizer._additional_special_tokens = new_ast
    tokenizer.special_tokens_map["additional_special_tokens"] = new_ast

    ###End
    
    patch_model(model_type=model.config.model_type, mode="ring")
    console.log("Model patched with ring attention")
    device = accelerator.device
    config = model.config
    model = accelerator.prepare(model)
    
    # past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
    # loader = DataLoader(te_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    save_dir = os.path.join(args.gen_dir, f"{args.name}.json")
    if os.path.exists(save_dir):
        console.log(f"Resuming from {save_dir}")
        with open(save_dir, "r", encoding="utf-8") as f:
            generated_text = json.load(f)
    else:
        generated_text = {}

    console.log(save_dir)

    pending = []
    for idx, uid in te_dataset.index_to_key_dict.items():
        if uid not in generated_text:
            _, batch = te_dataset[idx]
            pending.append((uid, batch))
    console.log(f"{len(pending)} / {len(te_dataset)} samples to test")

    # console.log(f"Test data: {len(te_dataset)} data points")
    # console.log("Testing...")

    with Progress(
        SpinnerColumn(),  # Shows a spinner
        TextColumn(
            "[progress.description]{task.description}"
        ),  # Displays additional info
        BarColumn(),  # Displays a progress bar
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # Shows percentage
    ) as progress:
        test_task = progress.add_task("Testing...", total=len(pending))
        with torch.no_grad():
            # generated_text = {}
            # time_list = []
            for step, (uuid, batch) in enumerate(pending):
                accelerator.wait_for_everyone()
                start_time = time.time()
                # uuid, batch = batch_data
                # batch_size = batch["input"]["input_ids"].size(0)

                # Process each sample in the batch as a micro-batch.
                # for i in range(batch_size):
                batch_input = batch["input"].copy()
                if "token_type_ids" in batch_input:
                    batch_input.pop("token_type_ids")
                micro_input = {
                    "input_ids": batch_input["input_ids"].to(device),
                    "attention_mask": batch_input["attention_mask"].to(device),
                    "labels": None,
                }

                accelerator.wait_for_everyone()
                # if "graph" in args.baseline_prompt:
                #     graph = batch["graph"]
                #     for key in model.gnn.type_of_graph:
                #         if key in graph.keys():
                #             graph[key] = graph[key].to(device)

                #     graph_mask = batch["graph_mask"].to(device)

                #     graph_token_index = torch.where(
                #         micro_input["input_ids"] == model.config.graph_token_id[1]
                #     )[1].tolist()

                if "graph" in args.baseline_prompt:
                    # if args.debug:
                    #     console.log("=" * 10 + "\n")
                    #     console.log(
                    #         f"Step {global_step}: Graph found in batch, checking keys..."
                    #     )
                    #     check_graph_exist_dict = {}
                    #     for key in GRAPH_KEYS:
                    #         check_graph_exist_dict[key] = False
                            
                    graph = batch["graph"]
                    for key in GRAPH_KEYS:
                        if key in graph.keys():
                            graph[key] = graph[key].to(device)
                            # if args.debug:
                            #     check_graph_exist_dict[key] = True
                    graph_mask = batch["graph_mask"].to(device)
                    graph_token_index = torch.where(
                        micro_input["input_ids"] == config.graph_token_id[1]
                    )[1].tolist()
                else:
                    graph = None
                    graph_mask = None
                    graph_token_index = None
                    # if args.debug:
                    #     console.log(f"Graph token id: {graph_token_index}")


                accelerator.wait_for_everyone()
                
                if accelerator.is_main_process:
                    outputs = model.module.generate(
                        inputs=micro_input["input_ids"],
                        graph=graph,
                        graph_mask=graph_mask,
                        graph_token_index=graph_token_index,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=False,
                    )
                    # else:
                    #     graph_token_index = None
                    #     outputs = model.generate(
                    #         inputs=micro_input["input_ids"],
                    #         graph_token_index=graph_token_index,
                    #         max_new_tokens=args.max_new_tokens,
                    #         do_sample=False,
                    #         use_cache=False,
                    #     )
    
                    out_text = tokenizer.batch_decode(outputs[:,micro_input["input_ids"].size(1):], skip_special_tokens=True)[0]
                    print(out_text)
                    # exit()
                    generated_text[uuid] = out_text
                    del outputs, micro_input, graph, graph_mask
    
                    if step % 2 == 0 or step == len(pending) - 1:
                        with open(save_dir, "w", encoding="utf-8") as f:
                            json.dump(generated_text, f, ensure_ascii=False, indent=4)
    
                    elapsed = time.time() - start_time
                    progress.update(
                        test_task,
                        advance=1,
                        description=f"Testing {step+1}/{len(pending)} — {elapsed:.2f}s",
                    )
                    accelerator.wait_for_everyone()

    console.log("Testing finished.")
    save_dir = os.path.join(args.gen_dir, f"{args.name}.json")
    with console.status("Saving results..."):
        # save generated text to jsonl file
        with open(save_dir, "w", encoding="utf-8") as f:
            # save as json file
            json.dump(generated_text, f, ensure_ascii=False, indent=4)

    console.log(f"Results saved to {save_dir}")


def validate(
    args: Namespace,
    loader: DataLoader,
    model: GLMFModelForCausalLM,
    config: GLMFModelConfig,
    device: torch.device,
    progress: Progress,
    accelerator: Accelerator,
):
    model.eval()
    if config is None:
        config = model.config
    with torch.no_grad():

        val_loss = 0.0
        num_item = 0

        if accelerator.is_main_process:
            val_task = progress.add_task("Validating...", total=len(loader))

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
                        "attention_mask": batch_input["attention_mask"][i].to(device),
                        "labels": batch_input["labels"][i].to(device),
                    }

                    if "graph" in args.baseline_prompt:
                        graph = batch["graph"][i]
                        for key in GRAPH_KEYS:
                            if key in graph.keys():
                                graph[key] = graph[key].to(device)

                        graph_mask = batch["graph_mask"][i].to(device)
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == config.graph_token_id[1]
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
                    if "graph" in args.baseline_prompt:
                        for key in GRAPH_KEYS:
                            if key in graph.keys():
                                graph[key] = graph[key].to("cpu")
                                graph.pop(key, None)
                        del graph_mask, graph
                    del outputs, loss, micro_input
                    gc.collect()
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError as e:
                tqdm.write(
                    f"OOM in batch {step}: input_dis {micro_input['input_ids'].size()} - graph_mask {graph_mask.size()}"
                )
                torch.cuda.empty_cache()
                continue

            if accelerator.is_main_process:
                progress.update(
                    val_task,
                    advance=1,
                    description=f"Batch {step + 1}/{len(loader)}: loss = {batch_loss/num_item:.4f}",
                )

            val_loss += batch_loss

        val_loss /= num_item
        return val_loss
        # model.train()

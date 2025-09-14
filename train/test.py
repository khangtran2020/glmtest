import os
import gc
import json
import time
import torch

# from torch.cuda.amp import autocast
import torch.nn.functional as F
from itertools import islice
from model.gnn import GRAPH_KEYS
from tqdm import tqdm
from rich import print as pprint
from data.loader import GLMFDataset, collate_fn
from model.model import GLMFModelForCausalLM, GLMFModelConfig
from transformers import PreTrainedTokenizer
from transformers import DynamicCache
from utils.constant import FUZZ_START_TOKEN, FUZZ_END_TOKEN

# from transformers import SinkCache
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from utils.utils import calculate_codebleu
from train.utils import (
    patch_model,
    revert_model_patch,
    extract_code_block,
)
from torch.utils.data import DataLoader
import torch.distributed as dist
from accelerate import Accelerator

# typing
from argparse import Namespace
from rich.console import Console
from typing import Optional


def test(
    args: Namespace,
    dataset: GLMFDataset,
    model: GLMFModelForCausalLM,
    console: Console,
    config: GLMFModelConfig = None,
    mixed_precision: str = "bf16",
):

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        log_with="wandb",
        project_dir=args.log_dir,
    )
    process_group = dist.group.WORLD
    if config is None:
        config = model.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    console.log("Testing on device ... :", device)
    if args.test_on_train:
        te_dataset = GLMFDataset(
            data=dict(islice(dataset.train_data.items(), 10)),
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
            n_hops=dataset.n_hops,
            testing=True,
            dtype=args.dtype,
            num_gpus=args.num_gpu,
        )
    else:
        te_mod_dataset = GLMFDataset(
            data=dataset.test_data["module"],
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
            n_hops=dataset.n_hops,
            testing=True,
            dtype=args.dtype,
            num_gpus=args.num_gpu,
        )
        te_proj_dataset = GLMFDataset(
            data=dataset.test_data["project"],
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
            n_hops=dataset.n_hops,
            testing=True,
            dtype=args.dtype,
            num_gpus=args.num_gpu,
        )
    tokenizer = dataset.llm_tokenizer
    if args.test_on_train:
        console.log(
            f"Test data: by project - {len(te_dataset)} data points (train set)"
        )
        console.log("Testing...")
    else:
        console.log(
            f"Test data: by project - {len(te_proj_dataset)} data points, by module - {len(te_mod_dataset)} data points"
        )
        console.log("Testing...")

    if args.test_on_train:
        with Progress(
            SpinnerColumn(),  # Shows a spinner
            TextColumn(
                "[progress.description]{task.description}"
            ),  # Displays additional info
            BarColumn(),  # Displays a progress bar
            TextColumn(
                "[progress.percentage]{task.percentage:>3.0f}%"
            ),  # Shows percentage
        ) as progress:
            test_task = progress.add_task(
                "Testing on train data...", total=len(te_dataset)
            )
            with torch.no_grad():
                generated_text = {}
                time_list = []
                for idx in range(len(te_dataset)):
                    start_time = time.time()
                    uuid, batch = te_dataset[idx]
                    console.log(f"Testing {uuid} - {idx}/{len(te_dataset)}")
                    batch_input = batch["input"].copy()
                    if "token_type_ids" in batch_input:
                        batch_input.pop("token_type_ids")

                    if args.debug and accelerator.is_main_process:
                        console.log(
                            f"[yellow]================ Example data point ================[/yellow]\n {batch['text']}\n\n[yellow]================ End of example data point ================[/yellow]"
                        )
                        console.log(
                            f"[yellow]================ Example tokenized ================[/yellow]\n {batch_input['input_ids'].squeeze(0).tolist()}\n\n[yellow]================ End of example tokenized ================[/yellow]"
                        )
                    micro_input = {
                        "input_ids": batch_input["input_ids"].to(device),
                        "attention_mask": batch_input["attention_mask"].to(device),
                        "labels": None,
                    }
                    if "graph" in args.baseline_prompt:
                        graph = batch["graph"]
                        for key in GRAPH_KEYS:
                            if key in graph.keys():
                                graph[key] = graph[key].to(device)
                        graph_mask = batch["graph_mask"].to(device)
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == config.graph_token_id[1]
                        )[1].tolist()
                    else:
                        graph = None
                        graph_mask = None
                        graph_token_index = None

                    inputs_embeds = model.extract_embedding(
                        input_ids=micro_input["input_ids"],
                        graph=graph,
                        inputs_embeds=None,
                        graph_mask=graph_mask,
                        graph_token_index=graph_token_index,
                    )

                    if args.debug and accelerator.is_main_process:
                        console.log(
                            f"Inputs embeds shape: {inputs_embeds.shape} | Graph token index: {len(graph_token_index)}"
                        )

                    if args.num_gpu == 1:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            if args.fuzz_model:
                                outputs = generate_fuzz(
                                    inputs_ids=micro_input["input_ids"],
                                    inputs_embeds=inputs_embeds,
                                    model=model,
                                    temperature=args.temp,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    accelerator=accelerator,
                                    tokenizer=dataset.llm_tokenizer,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=False,
                                    max_seq_len=args.max_seq_length,
                                )
                            else:
                                outputs = model.generate(
                                    inputs=micro_input["input_ids"],
                                    graph=graph,
                                    graph_mask=graph_mask,
                                    graph_token_index=graph_token_index,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=False,
                                    use_cache=True,
                                )
                        out_text = tokenizer.batch_decode(
                            outputs[:, micro_input["input_ids"].size(1) :],
                            skip_special_tokens=False if args.data_fuzz else True,
                        )[0]
                        # print(f"Generated text - {uuid}: {out_text}")
                        if args.debug and accelerator.is_main_process:
                            console.log(
                                f"\n\n[green]Generated text - {uuid} - num out tokens: {outputs[:, micro_input['input_ids'].size(1) :].size(1)}[/green]: {out_text}\n\n"
                            )

                        generated_text[uuid] = out_text
                        end_time = time.time()
                        process_time = end_time - start_time
                        time_list.append(process_time)
                        avg_time = sum(time_list) / len(time_list)
                        progress.update(
                            test_task,
                            advance=1,
                            description=f"Testing... {idx}/{len(te_dataset)} - {avg_time:.2f}s for 1 sample",
                        )
                    else:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = generate(
                                inputs_ids=micro_input["input_ids"],
                                inputs_embeds=inputs_embeds,
                                model=model,
                                temperature=args.temp,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                accelerator=accelerator,
                                tokenizer=dataset.llm_tokenizer,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                max_seq_len=args.max_seq_length,
                                console=console,
                            )

                        if accelerator.is_main_process:
                            out_text = tokenizer.batch_decode(
                                outputs[:, micro_input["input_ids"].size(1) :],
                                skip_special_tokens=False if args.data_fuzz else True,
                            )[0]

                            console.log(
                                f"\n\n[green]Generated text - {uuid} - num out tokens: {outputs[:, micro_input['input_ids'].size(1) :].size(1)}[/green]: {out_text}\n\n"
                            )

                            generated_text[uuid] = out_text
                            end_time = time.time()
                            process_time = end_time - start_time
                            time_list.append(process_time)
                            avg_time = sum(time_list) / len(time_list)
                            progress.update(
                                test_task,
                                advance=1,
                                description=f"Testing... {idx}/{len(te_dataset)} - {avg_time:.2f}s for 1 sample",
                            )
        if args.num_gpu == 1:
            console.log("Done Testing on train dataset finished.")
            save_dir = os.path.join(args.gen_dir, f"{args.name}_ontrain.json")
            with console.status("Saving results..."):
                # save generated text to jsonl file
                with open(save_dir, "w", encoding="utf-8") as f:
                    # save as json file
                    json.dump(generated_text, f, ensure_ascii=False, indent=4)
        else:
            console.log(
                "Done Testing on train dataset finished. Results saved in the main process only."
            )
            save_dir = os.path.join(args.gen_dir, f"{args.name}_ontrain.json")
            if accelerator.is_main_process:
                with console.status("Saving results..."):
                    # save generated text to jsonl file
                    with open(save_dir, "w", encoding="utf-8") as f:
                        # save as json file
                        json.dump(generated_text, f, ensure_ascii=False, indent=4)
    else:
        # Test projects
        with Progress(
            SpinnerColumn(),  # Shows a spinner
            TextColumn(
                "[progress.description]{task.description}"
            ),  # Displays additional info
            BarColumn(),  # Displays a progress bar
            TextColumn(
                "[progress.percentage]{task.percentage:>3.0f}%"
            ),  # Shows percentage
        ) as progress:
            test_task = progress.add_task(
                "Testing project levels...", total=len(te_proj_dataset)
            )
            with torch.no_grad():
                generated_text = {}
                time_list = []
                for idx in range(len(te_proj_dataset)):
                    start_time = time.time()
                    uuid, batch = te_proj_dataset[idx]
                    console.log(f"Testing {uuid} - {idx}/{len(te_proj_dataset)}")
                    batch_input = batch["input"].copy()
                    if "token_type_ids" in batch_input:
                        batch_input.pop("token_type_ids")

                    if args.debug and accelerator.is_main_process:
                        console.log(
                            f"[yellow]================ Example data point ================[/yellow]\n {batch['text']}\n\n[yellow]================ End of example data point ================[/yellow]"
                        )
                        console.log(
                            f"[yellow]================ Example tokenized ================[/yellow]\n {batch_input['input_ids'].squeeze(0).tolist()}\n\n[yellow]================ End of example tokenized ================[/yellow]"
                        )
                    micro_input = {
                        "input_ids": batch_input["input_ids"].to(device),
                        "attention_mask": batch_input["attention_mask"].to(device),
                        "labels": None,
                    }
                    for key in micro_input:
                        if micro_input[key] is not None:
                            print(f"Key: {key}, Dtype: {micro_input[key].dtype}")

                    if "graph" in args.baseline_prompt:
                        graph = batch["graph"]
                        for key in GRAPH_KEYS:
                            if key in graph.keys():
                                graph[key] = graph[key].to(device)
                        graph_mask = batch["graph_mask"].to(device)
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == config.graph_token_id[1]
                        )[1].tolist()
                    else:
                        graph = None
                        graph_mask = None
                        graph_token_index = None

                    inputs_embeds = model.extract_embedding(
                        input_ids=micro_input["input_ids"],
                        graph=graph,
                        inputs_embeds=None,
                        graph_mask=graph_mask,
                        graph_token_index=graph_token_index,
                    )

                    console.log(
                        f"Inputs embeds shape: {inputs_embeds.shape} | Graph token index: {len(graph_token_index)}"
                    )

                    if args.num_gpu == 1:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            if args.fuzz_model:
                                outputs = generate_fuzz(
                                    inputs_ids=micro_input["input_ids"],
                                    inputs_embeds=inputs_embeds,
                                    model=model,
                                    temperature=args.temp,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    accelerator=accelerator,
                                    tokenizer=dataset.llm_tokenizer,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=False,
                                    max_seq_len=args.max_seq_length,
                                )
                            else:
                                outputs = model.generate(
                                    inputs=micro_input["input_ids"],
                                    graph=graph,
                                    graph_mask=graph_mask,
                                    graph_token_index=graph_token_index,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=False,
                                    use_cache=True,
                                )
                        out_text = tokenizer.batch_decode(
                            outputs[:, micro_input["input_ids"].size(1) :],
                            skip_special_tokens=False if args.data_fuzz else True,
                        )[0]

                        console.log(
                            f"[green]Generated text - {uuid} - num out tokens: {outputs[:, micro_input['input_ids'].size(1) :].size(1)}[/green]: {out_text}"
                        )

                        generated_text[uuid] = out_text
                        end_time = time.time()
                        process_time = end_time - start_time
                        time_list.append(process_time)
                        avg_time = sum(time_list) / len(time_list)
                        progress.update(
                            test_task,
                            advance=1,
                            description=f"Testing... {idx}/{len(te_proj_dataset)} - {avg_time:.2f}s for 1 sample",
                        )
                    else:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = generate(
                                inputs_ids=micro_input["input_ids"],
                                inputs_embeds=inputs_embeds,
                                model=model,
                                temperature=args.temp,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                accelerator=accelerator,
                                tokenizer=dataset.llm_tokenizer,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                max_seq_len=args.max_seq_length,
                                console=console,
                            )

                        if accelerator.is_main_process:
                            out_text = tokenizer.batch_decode(
                                outputs[:, micro_input["input_ids"].size(1) :],
                                skip_special_tokens=False if args.data_fuzz else True,
                            )[0]

                            console.log(
                                f"[green]Generated text - {uuid} - num out tokens: {outputs[:, micro_input['input_ids'].size(1) :].size(1)}[/green]: {out_text}"
                            )

                            generated_text[uuid] = out_text
                            end_time = time.time()
                            process_time = end_time - start_time
                            time_list.append(process_time)
                            avg_time = sum(time_list) / len(time_list)
                            progress.update(
                                test_task,
                                advance=1,
                                description=f"Testing... {idx}/{len(te_proj_dataset)} - {avg_time:.2f}s for 1 sample",
                            )

        if args.num_gpu == 1:
            console.log("Done Testing Project level finished.")
            save_dir = os.path.join(args.gen_dir, f"{args.name}_proj.json")
            with console.status("Saving results..."):
                # save generated text to jsonl file
                with open(save_dir, "w", encoding="utf-8") as f:
                    # save as json file
                    json.dump(generated_text, f, ensure_ascii=False, indent=4)
        else:
            console.log(
                "Done Testing Project level finished. Results saved in the main process only."
            )
            save_dir = os.path.join(args.gen_dir, f"{args.name}_proj.json")
            if accelerator.is_main_process:
                with console.status("Saving results..."):
                    # save generated text to jsonl file
                    with open(save_dir, "w", encoding="utf-8") as f:
                        # save as json file
                        json.dump(generated_text, f, ensure_ascii=False, indent=4)

        # Test modules
        with Progress(
            SpinnerColumn(),  # Shows a spinner
            TextColumn(
                "[progress.description]{task.description}"
            ),  # Displays additional info
            BarColumn(),  # Displays a progress bar
            TextColumn(
                "[progress.percentage]{task.percentage:>3.0f}%"
            ),  # Shows percentage
        ) as progress:
            test_task = progress.add_task(
                "Testing modules levels...", total=len(te_mod_dataset)
            )
            with torch.no_grad():
                generated_text = {}
                time_list = []
                for idx in range(len(te_mod_dataset)):
                    start_time = time.time()
                    uuid, batch = te_mod_dataset[idx]
                    console.log(f"Testing {uuid} - {idx}/{len(te_mod_dataset)}")
                    batch_input = batch["input"].copy()
                    if "token_type_ids" in batch_input:
                        batch_input.pop("token_type_ids")
                    micro_input = {
                        "input_ids": batch_input["input_ids"].to(device),
                        "attention_mask": batch_input["attention_mask"].to(device),
                        "labels": None,
                    }
                    for key in micro_input:
                        if micro_input[key] is not None:
                            print(f"Key: {key}, Dtype: {micro_input[key].dtype}")

                    if "graph" in args.baseline_prompt:
                        graph = batch["graph"]
                        for key in GRAPH_KEYS:
                            if key in graph.keys():
                                graph[key] = graph[key].to(device)

                        graph_mask = batch["graph_mask"].to(device)
                        graph_token_index = torch.where(
                            micro_input["input_ids"] == config.graph_token_id[1]
                        )[1].tolist()
                    else:
                        graph = None
                        graph_mask = None
                        graph_token_index = None

                    inputs_embeds = model.extract_embedding(
                        input_ids=micro_input["input_ids"],
                        graph=graph,
                        inputs_embeds=None,
                        graph_mask=graph_mask,
                        graph_token_index=graph_token_index,
                    )

                    console.log(
                        f"Inputs embeds shape: {inputs_embeds.shape} | Graph token index: {len(graph_token_index)}"
                    )

                    if args.num_gpu == 1:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            if args.fuzz_model:
                                outputs = generate_fuzz(
                                    inputs_ids=micro_input["input_ids"],
                                    inputs_embeds=inputs_embeds,
                                    model=model,
                                    temperature=args.temp,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    accelerator=accelerator,
                                    tokenizer=dataset.llm_tokenizer,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=False,
                                    max_seq_len=args.max_seq_length,
                                )
                            else:
                                outputs = model.generate(
                                    inputs=micro_input["input_ids"],
                                    graph=graph,
                                    graph_mask=graph_mask,
                                    graph_token_index=graph_token_index,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=False,
                                    use_cache=True,
                                )
                        out_text = tokenizer.batch_decode(
                            outputs[:, micro_input["input_ids"].size(1) :],
                            skip_special_tokens=False if args.fuzz_model else True,
                        )[0]

                        # print(f"Generated text - {uuid}: {out_text}")
                        console.log(
                            f"[green]Generated text - {uuid} - num out tokens: {outputs[:, micro_input['input_ids'].size(1) :].size(1)}[/green]: {out_text}"
                        )

                        generated_text[uuid] = out_text
                        end_time = time.time()
                        process_time = end_time - start_time
                        time_list.append(process_time)
                        avg_time = sum(time_list) / len(time_list)
                        progress.update(
                            test_task,
                            advance=1,
                            description=f"Testing... {idx}/{len(te_proj_dataset)} - {avg_time:.2f}s for 1 sample",
                        )
                    else:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = generate(
                                inputs_ids=micro_input["input_ids"],
                                inputs_embeds=inputs_embeds,
                                model=model,
                                temperature=args.temp,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                accelerator=accelerator,
                                tokenizer=dataset.llm_tokenizer,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                max_seq_len=args.max_seq_length,
                                console=console,
                                process_group=process_group,
                            )

                        if accelerator.is_main_process:
                            out_text = tokenizer.batch_decode(
                                outputs[:, micro_input["input_ids"].size(1) :],
                                skip_special_tokens=False if args.fuzz_model else True,
                            )[0]

                            console.log(
                                f"[green]Generated text - {uuid} - num out tokens: {outputs[:, micro_input['input_ids'].size(1) :].size(1)}[/green]: {out_text}"
                            )

                            generated_text[uuid] = out_text
                            end_time = time.time()
                            process_time = end_time - start_time
                            time_list.append(process_time)
                            avg_time = sum(time_list) / len(time_list)
                            progress.update(
                                test_task,
                                advance=1,
                                description=f"Testing... {idx}/{len(te_mod_dataset)} - {avg_time:.2f}s for 1 sample",
                            )

        if args.num_gpu == 1:
            console.log("Done Testing Module level finished.")
            save_dir = os.path.join(args.gen_dir, f"{args.name}_mod.json")
            with console.status("Saving results..."):
                # save generated text to jsonl file
                with open(save_dir, "w", encoding="utf-8") as f:
                    # save as json file
                    json.dump(generated_text, f, ensure_ascii=False, indent=4)
        else:
            console.log(
                "Done Testing Module level finished. Results saved in the main process only."
            )
            save_dir = os.path.join(args.gen_dir, f"{args.name}_mod.json")
            if accelerator.is_main_process:
                with console.status("Saving results..."):
                    # save generated text to jsonl file
                    with open(save_dir, "w", encoding="utf-8") as f:
                        # save as json file
                        json.dump(generated_text, f, ensure_ascii=False, indent=4)


def eval_bleu_score(
    args: Namespace,
    dataset: GLMFDataset,
    console: Console,
):
    bleu = 0
    codeBleu = 0
    data = dataset.test_data
    with open(args.gen_file_path, "r", encoding="utf-8") as f:
        generate_response = json.load(f)

    i = 0
    for key in data.keys():
        if key not in generate_response.keys():
            continue
        else:
            i += 1
            file_path = data[key]
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    ground_truth = json.load(f)
            except FileNotFoundError:
                print(f"Error: File not found: {file_path}")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            ref = extract_code_block(ground_truth["response"])
            pred = extract_code_block(generate_response[key])
            result = calculate_codebleu(ref, pred)
            bleu += result["bleu_score"]
            codeBleu += result["codebleu_score"]
    console.log(f"[green]Bleu Score: {bleu/i}[/green]")
    console.log(f"[green]CodeBleu Score: {codeBleu/i}[/green]")


def validate(
    args: Namespace,
    loader: DataLoader,
    model: GLMFModelForCausalLM,
    config: GLMFModelConfig,
    device: torch.device,
    progress: Progress,
    console: Console,
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
                        graph = None
                        graph_mask = None

                    outputs = model(
                        **micro_input,
                        graph=graph,
                        graph_mask=graph_mask,
                        graph_token_index=graph_token_index,
                    )
                    loss = outputs.loss
                    all_losses = accelerator.gather(loss)
                    all_losses = torch.where(torch.isnan(all_losses), 0.0, all_losses)
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
                    loss = loss.to("cpu")
                    # for key in outputs.keys():
                    #     outputs[key] = outputs[key].to("cpu")
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

        if accelerator.is_main_process:
            progress.update(val_task, visible=False)
        val_loss /= num_item
        return val_loss
        # model.train()


def generate(
    inputs_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    model: GLMFModelForCausalLM,
    temperature: float,
    top_k: int,
    top_p: float,
    accelerator: Accelerator,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int,
    do_sample: bool = False,
    console: Console = None,
    process_group=None,
    max_seq_len: Optional[int] = None,
):
    position_ids = (
        torch.arange(inputs_embeds.shape[1])
        .unsqueeze(0)
        .expand(inputs_embeds.shape[0], -1)
    )
    original_attn_dict = patch_model(process_group=process_group)
    batch_size = inputs_embeds.shape[0]
    device = inputs_embeds.device

    # Keep track of which sequences are finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    past_key_values = DynamicCache()
    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )

    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )

    generated_ids = inputs_ids.clone()

    current_length = inputs_embeds.shape[1]

    model.eval()
    with torch.inference_mode():

        for step in range(max_new_tokens):

            if current_length >= max_seq_len:
                break

            if step == 0:

                positional_embedding = model.llm_model.model.rotary_emb(
                    inputs_embeds, position_ids.to(inputs_embeds.device)
                )

                outputs = model.forward_llm(
                    inputs_embeds=inputs_embeds, position_ids=position_ids
                )
                logits = outputs.logits
                out_past_key_values = outputs.past_key_values
                preds = logits_to_prediction(
                    logits, temperature, top_k, top_p, do_sample
                )

                def undo_extract_local(gathered_value, world_size, dim=1):
                    value_chunks = gathered_value.chunk(2 * world_size, dim=dim)
                    reordered_chunks = [None] * (2 * world_size)
                    for i in range(world_size):
                        reordered_chunks[i] = value_chunks[i * 2]
                        reordered_chunks[2 * world_size - i - 1] = value_chunks[
                            i * 2 + 1
                        ]
                    return torch.cat(reordered_chunks, dim=dim)

                gathered_logits = accelerator.gather(preds.squeeze(0)).unsqueeze(0)
                pred = undo_extract_local(gathered_logits, accelerator.num_processes)
                pred = pred[:, current_length - 1 : current_length]
                past_key_values = merge_sequence_parallel_cache_optimized(
                    cache=past_key_values,
                    local_outcome=out_past_key_values,
                    cache_position=cache_position,
                    positional_embedding=positional_embedding,
                    accelerator=accelerator,
                )
            else:
                if accelerator.is_main_process:
                    if step == 1:
                        revert_model_patch(original_methods=original_attn_dict)

                    generated_embeddings = model.extract_embedding(
                        input_ids=generated_ids[:, current_length - 1],
                        graph=None,
                        graph_mask=None,
                        graph_token_index=None,
                    ).unsqueeze(0)

                    outputs = model.llm_model.forward(
                        inputs_embeds=generated_embeddings,
                        past_key_values=past_key_values,
                        position_ids=None,
                    )
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values

                    preds = logits_to_prediction(
                        logits, temperature, top_k, top_p, do_sample
                    )
                else:
                    # clean up everything and prepare for the next step to release RAM
                    # move to cpu first
                    if step == 1:
                        inputs_embeds = inputs_embeds.cpu()
                        position_ids = position_ids.cpu()
                        logits = logits.cpu()

                        del inputs_embeds, position_ids, logits, outputs
                        gc.collect()
                        torch.cuda.empty_cache()

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                pred = pred.masked_fill(finished, tokenizer.pad_token_id)
                # print(
                #     f"Rank {model.rank} - Step {step + 1}/{max_new_tokens} - Pred shape: {pred.shape} - Finished: {generated_ids.shape}"
                # )
                generated_ids = torch.cat([generated_ids, pred], dim=1)

                finished = finished | (pred[:, -1] == tokenizer.eos_token_id)
                current_length += 1
                if finished.all():
                    break
    if accelerator.is_main_process:
        return generated_ids
    else:
        # If not the main process, return None
        return None


def logits_to_prediction(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float, do_sample: bool
):

    # Apply temperature scaling
    if temperature != 0.0:
        logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None:
        logits = _top_k_filtering(logits, top_k)

    # Apply top-p (nucleus) filtering

    # print(f"Using argmax for prediction: {logits.shape} - {logits}")

    if do_sample:
        probs = F.softmax(logits, dim=-1)
        if top_p is not None:
            probs = _top_p_filtering(probs, top_p)
        preds = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        preds = torch.argmax(logits, dim=-1)

    return preds


def _top_k_filtering(logits, top_k):
    """Filter logits to keep only top k tokens"""
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = -float("inf")
    return logits


def _top_p_filtering(logits, top_p):
    """Filter logits using nucleus (top-p) sampling"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = -float("inf")
    return logits


def merge_sequence_parallel_cache_optimized(
    cache: DynamicCache,
    local_outcome: tuple,
    cache_position: torch.Tensor,
    positional_embedding: tuple,
    accelerator: Accelerator,
):
    """
    Alternative implementation that's more memory efficient for very long sequences.
    Only gathers when actually needed and can work with chunked processing.
    """

    # print(f"Info of local_cache: {local_cache}")

    if accelerator.num_processes == 1:
        return cache

    for layer_idx in range(len(local_outcome)):
        k_local, v_local = local_outcome[layer_idx]  # [B, H, L_local, D]

        # Use accelerator's built-in gather - this handles the distributed communication
        k_all = accelerator.gather(k_local)  # [B*world_size, H, L_local, D]
        v_all = accelerator.gather(v_local)

        B, H, L_local, D = k_local.shape
        world_size = accelerator.num_processes

        # Concatenate along sequence dimension (dim=3 after reshaping)
        k_merged = torch.cat(
            [k_all[i] for i in range(world_size)], dim=1
        )  # [B, H, L_total, D]
        v_merged = torch.cat([v_all[i] for i in range(world_size)], dim=1)
        k_merged = k_merged.unsqueeze(0)  # [1, B*world_size, H, L_total, D]
        v_merged = v_merged.unsqueeze(0)  # [1, B*world_size, H, L_total, D]

        cos, sin = positional_embedding
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        _, _ = cache.update(k_merged, v_merged, layer_idx, cache_kwargs)

    return cache


def generate_fuzz(
    inputs_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    model: GLMFModelForCausalLM,
    temperature: float,
    top_k: int,
    top_p: float,
    accelerator: Accelerator,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int,
    do_sample: bool = False,
    max_seq_len: Optional[int] = None,
):

    batch_size = inputs_embeds.shape[0]
    device = inputs_embeds.device

    position_ids = (
        torch.arange(inputs_embeds.shape[1])
        .unsqueeze(0)
        .expand(inputs_embeds.shape[0], -1)
    ).to(device)

    # Keep track of which sequences are finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    past_key_values = DynamicCache()
    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    fuzz_start_id = tokenizer.convert_tokens_to_ids(FUZZ_START_TOKEN)
    fuzz_end_id = tokenizer.convert_tokens_to_ids(FUZZ_END_TOKEN)

    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )

    generated_ids = inputs_ids.clone()

    current_length = inputs_embeds.shape[1]
    fuzzing_mask = torch.zeros(inputs_ids.shape, device=inputs_ids.device)

    model.eval()
    with torch.inference_mode():

        for step in range(max_new_tokens):

            if current_length >= max_seq_len:
                break

            if step == 0:
                outputs = model.forward(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    fuzzing_mask=fuzzing_mask.unsqueeze(-1),
                    past_key_values=past_key_values,
                )
                logits = outputs.logits
                preds = logits_to_prediction(
                    logits, temperature, top_k, top_p, do_sample
                )
                pred = pred[:, current_length - 1 : current_length]
            else:
                if accelerator.is_main_process:

                    generated_embeddings = model.extract_embedding(
                        input_ids=generated_ids[:, current_length - 1],
                        graph=None,
                        graph_mask=None,
                        graph_token_index=None,
                    ).unsqueeze(0)

                    outputs = model.llm_model.forward(
                        inputs_embeds=generated_embeddings,
                        past_key_values=past_key_values,
                        position_ids=None,
                        fuzzing_mask=fuzzing_mask.unsqueeze(-1),
                    )
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values

                    preds = logits_to_prediction(
                        logits, temperature, top_k, top_p, do_sample
                    )
                    pred = preds[:, -1:].clone()

            pred = pred.masked_fill(finished, tokenizer.pad_token_id)
            generated_ids = torch.cat([generated_ids, pred], dim=1)

            # Update the fuzzing mask to not fuzz the generated token
            fuzzing_mask = torch.cat(
                [fuzzing_mask, torch.zeros((batch_size, 1), device=device)], dim=1
            )
            for i in range(generated_ids.shape[0]):
                saw_start = False
                for j in range(generated_ids.shape[1]):
                    if saw_start:
                        fuzzing_mask[i, j] = 1
                    if generated_ids[i, j] == fuzz_start_id:
                        saw_start = True
                    elif generated_ids[i, j] == fuzz_end_id:
                        saw_start = False

            finished = finished | (pred[:, -1] == tokenizer.eos_token_id)
            current_length += 1
            if finished.all():
                break

    return generated_ids

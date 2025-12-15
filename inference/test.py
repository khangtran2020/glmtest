import os
import gc
import json
import time
import torch
from itertools import islice
from rich import print as pprint
from data.core import Data
from data.loader import GLMFDataset, collate_fn
from model.model import GLMFModelForCausalLM, GLMFModelConfig, GLMFModelFuzzing
from transformers import PreTrainedTokenizer, GenerationConfig
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from utils.utils import calculate_codebleu
from train.utils import extract_code_block
from torch.utils.data import DataLoader
from accelerate import Accelerator
from functools import partial

# typing
from argparse import Namespace
from rich.console import Console


def test(
    args: Namespace,
    dataset: Data,
    model: GLMFModelForCausalLM,
    console: Console,
    config: GLMFModelConfig = None,
    mixed_precision: str = "bf16",
):
    collate_fn_ = partial(collate_fn, tokenizer=dataset.llm_tokenizer)
    tokenizer = dataset.llm_tokenizer
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
            metadata=model.metadata,
            num_gpus=args.num_gpu,
        )
        generate_and_save_on_one_dataset(
            dataset=te_dataset,
            model=model,
            args=args,
            console=console,
            device=device,
            config=config,
            tokenizer=tokenizer,
            collate_fn_=collate_fn_,
            # accelerator=accelerator,
            suffix="train",
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
            metadata=model.metadata,
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
            metadata=model.metadata,
            num_gpus=args.num_gpu,
        )
        generate_and_save_on_one_dataset(
            dataset=te_mod_dataset,
            model=model,
            args=args,
            console=console,
            device=device,
            config=config,
            tokenizer=tokenizer,
            collate_fn_=collate_fn_,
            # accelerator=accelerator,
            suffix="module",
        )
        generate_and_save_on_one_dataset(
            dataset=te_proj_dataset,
            model=model,
            args=args,
            console=console,
            device=device,
            config=config,
            tokenizer=tokenizer,
            collate_fn_=collate_fn_,
            # accelerator=accelerator,
            suffix="project",
        )


def generate_and_save_on_one_dataset(
    dataset,
    model,
    args: Namespace,
    console: Console,
    device: torch.device,
    tokenizer: PreTrainedTokenizer,
    collate_fn_: callable,
    dataloader: DataLoader = None,
    config: GLMFModelConfig = None,
    suffix: str = "train",
    do_save: bool = True,
):
    if dataloader is None:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_,
        )
    else:
        loader = dataloader

    generated_text = {}
    processed_instance_id = []
    if do_save:
        save_dir_text = os.path.join(args.gen_dir, f"{args.name}_text_{suffix}.jsonl")
        save_dir_code = os.path.join(args.gen_dir, f"{args.name}_code_{suffix}.jsonl")
        # create an empty file if not exists
        if os.path.exists(save_dir_text):
            with open(save_dir_text, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    instance = json.loads(line)
                    processed_instance_id.extend(list(instance.keys()))
                    generated_text.update(instance)

    with Progress(
        SpinnerColumn(),  # Shows a spinner
        TextColumn(
            "[progress.description]{task.description}"
        ),  # Displays additional info
        BarColumn(),  # Displays a progress bar
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # Shows percentage
    ) as progress:
        test_task = progress.add_task("Generating for one task ...", total=len(loader))
        with torch.no_grad():
            time_list = []
            for uuid, batch in loader:

                # skip already processed instances
                if do_save:
                    filtered_indices = [
                        i
                        for i, idx in enumerate(uuid)
                        if idx not in processed_instance_id
                    ]

                    if len(filtered_indices) == 0:
                        progress.update(test_task, advance=1)
                        continue

                    batch["text"] = [
                        text
                        for i, text in enumerate(batch["text"])
                        if i in filtered_indices
                    ]
                    batch["graph_mask"] = (
                        [
                            graph_mask
                            for i, graph_mask in enumerate(batch["graph_mask"])
                            if i in filtered_indices
                        ]
                        if batch["graph_mask"] is not None
                        else None
                    )
                    batch["graph"] = (
                        [
                            graph
                            for i, graph in enumerate(batch["graph"])
                            if i in filtered_indices
                        ]
                        if batch["graph"] is not None
                        else None
                    )
                    collated_input_filtered = {}
                    for key in batch["input"].keys():
                        item = batch["input"][key]
                        if isinstance(item, torch.Tensor):
                            collated_input_filtered[key] = item[filtered_indices]
                        else:
                            collated_input_filtered[key] = [
                                item[i] for i in filtered_indices
                            ]

                    uuid = [idx for i, idx in enumerate(uuid) if i in filtered_indices]

                start_time = time.time()
                batch_size = batch["input"]["input_ids"].size(0)
                if "token_type_ids" in batch["input"]:
                    batch["input"].pop("token_type_ids")

                micro_input = {
                    "input_ids": batch["input"]["input_ids"].to(device),
                    "attention_mask": batch["input"]["attention_mask"].to(device),
                    "labels": None,
                }
                if "graph" in args.baseline_prompt:
                    graphs = []
                    graph_masks = []
                    graph_token_indices = []

                    for i in range(batch_size):
                        graph = batch["graph"][i]
                        graph = graph.to(device)
                        # for key in GRAPH_KEYS:
                        #     if key in graph.keys():
                        #         graph[key] = graph[key].to(device)
                        #         graph[key].ndata["feat"] = (
                        #             graph[key].ndata["feat"].to(device)
                        #         )

                        graph_mask = [
                            mask.to(device) for mask in batch["graph_mask"][i]
                        ]
                        graph_token_index = torch.where(
                            micro_input["input_ids"][i] == config.graph_token_id[1]
                        )[0].tolist()
                        graphs.append(graph)
                        graph_masks.append(graph_mask)
                        graph_token_indices.append(graph_token_index)
                else:
                    graphs = None
                    graph_masks = None
                    graph_token_indices = None

                if args.num_gpu > 1:
                    inputs_embeds = model.module.extract_embedding(
                        input_ids=micro_input["input_ids"],
                        graphs=graphs,
                        inputs_embeds=None,
                        graph_masks=graph_masks,
                        graph_token_indices=graph_token_indices,
                    )
                else:
                    inputs_embeds = model.extract_embedding(
                        input_ids=micro_input["input_ids"],
                        graphs=graphs,
                        inputs_embeds=None,
                        graph_masks=graph_masks,
                        graph_token_indices=graph_token_indices,
                    )

                generation_config = GenerationConfig(
                    temperature=args.temp,
                    top_p=0.95,
                    top_k=40,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

                dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    if args.num_gpu > 1:
                        outputs = model.module.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=micro_input["attention_mask"],
                            generation_config=generation_config,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                        if isinstance(model, GLMFModelFuzzing):
                            model.module.clear_cache()
                    else:
                        outputs = model.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=micro_input["attention_mask"],
                            generation_config=generation_config,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        if isinstance(model, GLMFModelFuzzing):
                            model.clear_cache()

                out_text = tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=False if args.data_fuzz else True,
                )

                # Clear memory

                for key in micro_input.keys():
                    if micro_input[key] is not None:
                        micro_input[key] = micro_input[key].to("cpu")
                if "graph" in args.baseline_prompt:
                    for graph in graphs:
                        graph = graph.to("cpu")
                        # for key in GRAPH_KEYS:
                        #     if key in graph.keys():
                        #         graph[key] = graph[key].to("cpu")
                        #         graph.pop(key, None)
                    for graph_mask in graph_masks:
                        for mask in graph_mask:
                            mask = mask.to("cpu")
                    del graph_masks, graphs

                del outputs, micro_input
                gc.collect()
                torch.cuda.empty_cache()

                for i, idx in enumerate(uuid):
                    generated_text[idx] = out_text[i]

                if do_save:
                    with console.status("Saving results..."):
                        # append generated text to jsonl file
                        with open(save_dir_text, "a", encoding="utf-8") as f:
                            for i, idx in enumerate(uuid):
                                instance_data = {idx: out_text[i]}
                                f.write(
                                    json.dumps(instance_data, ensure_ascii=False) + "\n"
                                )

                        with open(save_dir_code, "a", encoding="utf-8") as f:
                            for i, idx in enumerate(uuid):
                                if extract_code_block(text=out_text[i]) == "":
                                    continue
                                instance_data = {
                                    idx: extract_code_block(text=out_text[i])
                                }
                                f.write(
                                    json.dumps(instance_data, ensure_ascii=False) + "\n"
                                )

                end_time = time.time()
                process_time = end_time - start_time
                time_list.append(process_time)
                avg_time = sum(time_list) / len(time_list)
                progress.update(
                    test_task,
                    advance=1,
                    description=f"Testing... - {avg_time:.2f}s for 1 batch",
                )

    console.log("Done Testing on train dataset finished.")
    return generated_text


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


def logging_gpu_usage(step: int, console: Console):
    torch.cuda.reset_peak_memory_stats()
    gpu_memory = torch.cuda.memory_allocated() / (1024**3)
    gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    gpu_free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    gpu_free = gpu_free / (1024**3)
    pprint(
        f"[blue]At step {step} - GPU memory allocated: {gpu_memory:.2f} GB, GPU memory reserved: {gpu_reserved:.2f} GB, Peak memory usage: {peak_memory:.2f}[/blue]"
    )


def validate(
    args: Namespace,
    loader: DataLoader,
    model: GLMFModelForCausalLM,
    config: GLMFModelConfig,
    progress: Progress,
    accelerator: Accelerator,
    multi_gpu: bool = False,
):
    model.eval()
    with torch.no_grad():

        val_loss = 0.0
        num_item = 0

        if accelerator.is_main_process and progress is not None:
            val_task = progress.add_task(
                "Validating...", total=len(loader), step_time=0.0
            )

        for step, batch in enumerate(loader):
            batch_loss = 0.0
            batch_size = batch["input"]["input_ids"].size(0)
            num_item += batch_size

            start_time = time.time()

            if "token_type_ids" in batch["input"]:
                batch["input"].pop("token_type_ids")

            micro_input = {
                "input_ids": batch["input"]["input_ids"],
                "attention_mask": batch["input"]["attention_mask"],
                "labels": batch["input"]["labels"],
            }

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

            try:
                outputs = model(
                    **micro_input,
                    graphs=graphs,
                    graph_masks=graph_masks,
                    graph_token_indices=graph_token_indices,
                )
            except Exception as e:
                print(f"RuntimeError during validation at step {step}: {e}")
                for graph in graphs:
                    print(graph.edge_index_dict)
                raise e

            loss = outputs.loss
            if multi_gpu:
                all_losses = accelerator.gather(loss)
                all_losses = torch.where(torch.isnan(all_losses), 0.0, all_losses)
                all_losses = all_losses.to("cpu").detach().float()
                total_loss = torch.sum(all_losses)
                batch_loss += total_loss.to("cpu").detach().float().item()
            else:
                batch_loss += loss.to("cpu").detach().float().item()

            # logging_gpu_usage(step=step, console=console)

            for key in micro_input.keys():
                micro_input[key] = micro_input[key].to("cpu")
            if "graph" in args.baseline_prompt:
                for graph in graphs:
                    graph = graph.to("cpu")
                for graph_mask in graph_masks:
                    for mask in graph_mask:
                        mask = mask.to("cpu")
                del graph_masks, graphs, graph_token_indices
            loss = loss.to("cpu")
            del outputs, loss, micro_input, batch
            gc.collect()
            torch.cuda.empty_cache()

            if accelerator.is_main_process and progress is not None:
                progress.update(
                    val_task,
                    advance=1,
                    step_time=time.time() - start_time,
                    description=f"Batch {step + 1}/{len(loader)}: loss = {batch_loss/num_item:.4f}",
                )

            val_loss += batch_loss

            # Periodic cache clearing (every 10 batches)
            if step % 10 == 0:
                torch.cuda.empty_cache()

        if accelerator.is_main_process and progress is not None:
            progress.update(val_task, visible=False)
        val_loss /= num_item
    model.train()
    return val_loss

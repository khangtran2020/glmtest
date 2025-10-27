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
from data.core import Data
from data.loader import GLMFDataset, collate_fn
from model.model import GLMFModelForCausalLM, GLMFModelConfig, GLMFModelFuzzing
from transformers import PreTrainedTokenizer, DynamicCache, GenerationConfig
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
from functools import partial

# typing
from argparse import Namespace
from rich.console import Console
from typing import Optional


def test(
    args: Namespace,
    dataset: Data,
    model: GLMFModelForCausalLM,
    console: Console,
    config: GLMFModelConfig = None,
    mixed_precision: str = "bf16",
):
    collate_fn_ = partial(
        collate_fn, tokenizer=dataset.llm_tokenizer, max_seq_length=args.max_seq_length
    )
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        log_with="wandb",
        project_dir=args.log_dir,
    )
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
            num_gpus=args.num_gpu,
        )
        generate_and_save_on_one_dataset(
            dataset=te_dataset,
            model=model,
            args=args,
            console=console,
            device=device,
            tokenizer=tokenizer,
            collate_fn_=collate_fn_,
            accelerator=accelerator,
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
        generate_and_save_on_one_dataset(
            dataset=te_mod_dataset,
            model=model,
            args=args,
            console=console,
            device=device,
            tokenizer=tokenizer,
            collate_fn_=collate_fn_,
            accelerator=accelerator,
            suffix="module",
        )
        generate_and_save_on_one_dataset(
            dataset=te_proj_dataset,
            model=model,
            args=args,
            console=console,
            device=device,
            tokenizer=tokenizer,
            collate_fn_=collate_fn_,
            accelerator=accelerator,
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
    accelerator: Accelerator,
    suffix: str = "train",
    do_save: bool = True,
):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_,
    )
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
            generated_text = {}
            time_list = []
            for uuid, batch in loader:
                start_time = time.time()
                batch_size = batch["input"]["input_ids"].size(0)
                if "token_type_ids" in batch["input"]:
                    batch["input"].pop("token_type_ids")

                if args.debug and accelerator.is_main_process:
                    console.log(
                        f"[yellow]================ Example data point ================[/yellow]\n {batch['text'][0]}\n\n[yellow]================ End of example data point ================[/yellow]"
                    )
                    console.log(
                        f"[yellow]================ Example tokenized ================[/yellow]\n {batch['input']['input_ids']}\n\n[yellow]================ End of example tokenized ================[/yellow]"
                    )
                    console.log(
                        f"[yellow]================ Example attention_mask ================[/yellow]\n {batch['input']['attention_mask']}\n\n[yellow]================ End of example tokenized ================[/yellow]"
                    )
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
                        for key in GRAPH_KEYS:
                            if key in graph.keys():
                                graph[key] = graph[key].to(device)
                                graph[key].ndata["feat"] = (
                                    graph[key].ndata["feat"].to(device)
                                )

                        graph_mask = [
                            mask.to(device) for mask in batch["graph_mask"][i]
                        ]
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

                inputs_embeds = model.extract_embedding(
                    input_ids=micro_input["input_ids"],
                    graphs=graphs,
                    inputs_embeds=None,
                    graph_masks=graph_masks,
                    graph_token_indices=graph_token_indices,
                )

                if args.debug and accelerator.is_main_process:
                    console.log(
                        f"Inputs embeds shape: {inputs_embeds.shape} | Graph token index: {len(graph_token_index)}"
                    )

                generation_config = GenerationConfig(
                    temperature=0.2,
                    top_p=0.95,
                    top_k=40,
                    no_repeat_ngram_size=4,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    repetition_penalty=1.5,
                )

                with torch.autocast(device_type="cuda", dtype=torch.float16):
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

                # print(f"Generated text - {uuid}: {out_text}")
                if args.debug and accelerator.is_main_process:
                    console.log(
                        f"\n\n[green]Generated text - {uuid} - num out tokens: {outputs.size(1)}[/green]: {out_text}\n\n"
                    )

                for i, idx in enumerate(uuid):
                    generated_text[idx] = extract_code_block(out_text[i])

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
    if do_save:
        save_dir = os.path.join(args.gen_dir, f"{args.name}_{suffix}.json")
        with console.status("Saving results..."):
            # save generated text to jsonl file
            with open(save_dir, "w", encoding="utf-8") as f:
                # save as json file
                json.dump(generated_text, f, ensure_ascii=False, indent=4)

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
    device: torch.device,
    progress: Progress,
    console: Console,
    accelerator: Accelerator,
    multi_gpu: bool = False,
):
    model.eval()
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

                outputs = model(
                    **micro_input,
                    graphs=graphs,
                    graph_masks=graph_masks,
                    graph_token_indices=graph_token_indices,
                )
                loss = outputs.loss
                if multi_gpu:
                    all_losses = accelerator.gather(loss)
                    all_losses = torch.where(torch.isnan(all_losses), 0.0, all_losses)
                    total_loss = torch.sum(all_losses)
                    batch_loss += total_loss.detach().float().item()
                else:
                    batch_loss += loss.detach().float().item()

                logging_gpu_usage(step=step, console=console)

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
                    del graph_masks, graphs
                loss = loss.to("cpu")
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
    finished = torch.zeros(batch_size, dtype=torch.bool, device="cpu")

    past_key_values = DynamicCache()
    fuzz_start_id = tokenizer.convert_tokens_to_ids(FUZZ_START_TOKEN)
    fuzz_end_id = tokenizer.convert_tokens_to_ids(FUZZ_END_TOKEN)

    generated_ids = inputs_ids.clone().to("cpu")

    current_length = inputs_embeds.shape[1]
    fuzzing_mask = torch.zeros(inputs_ids.shape, device="cpu")

    pprint(f"[green]Shape of fuzzing mask initialized: {fuzzing_mask.size()}[/green]")

    model.eval()
    with torch.inference_mode():

        for step in range(max_new_tokens):

            if current_length >= max_seq_len:
                break

            if step == 0:
                outputs = model.forward(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    fuzzing_mask=fuzzing_mask.unsqueeze(-1).to(device),
                    past_key_values=past_key_values,
                )
                logits = outputs.logits
                preds = logits_to_prediction(
                    logits, temperature, top_k, top_p, do_sample
                )
                pred = preds[:, current_length - 1 : current_length]
                # free memory
                preds = preds.cpu()
                inputs_embeds = inputs_embeds.cpu()
                position_ids = position_ids.cpu()
                logits = logits.cpu()

                del inputs_embeds, position_ids, logits, outputs, preds
                gc.collect()
                torch.cuda.empty_cache()
            else:
                if accelerator.is_main_process:

                    generated_embeddings = (
                        model.extract_embedding(
                            input_ids=generated_ids[:, current_length - 1],
                            graph=None,
                            graph_mask=None,
                            graph_token_index=None,
                        )
                        .unsqueeze(0)
                        .to(device)
                    )

                    outputs = model.llm_model.forward(
                        inputs_embeds=generated_embeddings,
                        past_key_values=past_key_values,
                        position_ids=None,
                        fuzzing_mask=fuzzing_mask.unsqueeze(-1).to(device),
                    )
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values

                    preds = logits_to_prediction(
                        logits, temperature, top_k, top_p, do_sample
                    )
                    pred = preds[:, -1:].clone()

                    # free memory
                    generated_embeddings = generated_embeddings.cpu()
                    logits = logits.cpu()
                    preds = preds.cpu()

                    del generated_embeddings, logits, outputs, preds
                    gc.collect()
                    torch.cuda.empty_cache()

            pred = pred.to("cpu").masked_fill(finished, tokenizer.pad_token_id)
            # pprint(f"Device of pred: {pred.device}")
            generated_ids = torch.cat([generated_ids, pred], dim=1).to("cpu")
            # pred = pred.to(device)
            del pred
            gc.collect()
            torch.cuda.empty_cache()

            # Update the fuzzing mask to not fuzz the generated token
            fuzzing_mask = torch.cat(
                [fuzzing_mask, torch.zeros((batch_size, 1), device="cpu")], dim=1
            )
            for i in range(generated_ids.shape[0]):
                saw_start = False
                for j in range(generated_ids.shape[1]):
                    if saw_start:
                        fuzzing_mask[i, j] = 1
                    if generated_ids[i, j] == fuzz_start_id:
                        pprint(f"[red]Found fuzzing start at position {j}[/red]")
                        saw_start = True
                    elif generated_ids[i, j] == fuzz_end_id:
                        pprint(f"[red]Found fuzzing end at position {j}[/red]")
                        saw_start = False

            pprint(f"[blue]Fuzzing mask updated: {fuzzing_mask.size()}[/blue]")

            finished = finished | (generated_ids[:, -1] == tokenizer.eos_token_id)
            current_length += 1
            if finished.all():
                break

    return generated_ids

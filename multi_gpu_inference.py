import os
import json
import torch
import warnings
from itertools import islice
from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
from graph.utils import get_graph
from model.model import get_model
from inference.test import generate_and_save_on_one_dataset
from data.loader import GLMFDataset, collate_fn
from functools import partial
from accelerate import Accelerator, PartialState
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore", category=UserWarning)


def test_with_accelerate(args):
    """Test model using Accelerate for automatic multi-GPU handling"""

    # Initialize Accelerator - this handles all distributed setup
    accelerator = Accelerator()

    console.log(
        f"[green]Process {accelerator.process_index} of {accelerator.num_processes} started[/green]"
    )
    console.log(f"[green]Using device: {accelerator.device}[/green]")

    # Initialize dataset (only on main process to avoid duplication)
    with accelerator.main_process_first():
        graph = get_graph(
            args=args,
            graph_type=args.graph_type,
            logger=console,
        )
        dataset = get_dataset(
            data_name=args.data,
            data_path=args.data_path,
            logger=console,
            feat_model=args.feat_model,
            llm_model=args.llm_model,
            max_pynguin_run_time=args.max_pynguin_run_time,
            docker_image=args.docker_image,
            num_cpu=args.num_cpu,
            graph=graph,
            llm_model_name=args.llm_model_name,
            model_name=args.model_name,
            data_max_length=args.max_seq_length,
            baseline_prompt=args.baseline_prompt,
            debug=args.debug,
            mode=args.mode,
            graph_sampling=args.graph_sampling,
            n_hops=args.n_layers,
            max_tokens=args.max_seq_length,
            raw_overwrite=args.raw_overwrite,
            repo=args.repo,
        )

        if dataset is None:
            console.log("[red]Dataset not found, exiting...[/red]")
            return

        if args.mode == "data":
            dataset.process_raw()
            console.log("Data processing completed. Exiting as mode is 'data'.")
            return

        if args.mode == "testgen":
            if args.module_path is None:
                dataset.prepare_data_for_test_gen()
        else:
            dataset.prepare_data()

        if args.repo is not None:
            dataset.prepare_data_by_repo()

        dataset.train_test_split(
            val_split=int(100), test_only=True if args.mode == "testgen" else False
        )

    # Wait for all processes to reach this point
    accelerator.wait_for_everyone()

    # Get model - Accelerate handles device placement
    model = get_model(
        args=args,
        tokenizer=dataset.llm_tokenizer,
        rank=accelerator.process_index,
        device=accelerator.device,
        console=console,
    )

    # Prepare model with Accelerate - this handles Flash Attention device assignment
    model = accelerator.prepare_model(model)

    # Unify dtype
    with accelerator.main_process_first():
        for n, p in model.named_parameters():
            if args.dtype == "bf16":
                if p.dtype != torch.bfloat16:
                    p.data = p.data.to(torch.bfloat16)
            elif args.dtype == "fp16":
                if p.dtype != torch.float16:
                    p.data = p.data.to(torch.float16)

    console.log(f"[Process {accelerator.process_index}] Model loaded successfully")

    # Prepare dataset
    if args.test_on_train:
        data = dict(islice(dataset.train_data.items(), 10))
    else:
        data = dataset.test_data["module"]

    te_dataset = GLMFDataset(
        data=data,
        tokenizer=dataset.llm_tokenizer,
        max_seq_length=args.max_seq_length,
        debug=args.debug,
        n_hops=dataset.n_hops,
        testing=True,
        dtype=args.dtype,
        num_gpus=args.num_gpu,
    )

    collate_fn_ = partial(
        collate_fn,
        tokenizer=dataset.llm_tokenizer,
        max_seq_length=args.max_seq_length,
    )

    # Create DataLoader - Accelerate will automatically shard the data
    dataloader = DataLoader(
        te_dataset,
        batch_size=1,
        collate_fn=collate_fn_,
        shuffle=False,
    )

    # Prepare dataloader with Accelerate
    dataloader = accelerator.prepare_data_loader(dataloader)

    # Run inference
    generate_and_save_on_one_dataset(
        dataset=te_dataset,
        model=model,
        args=args,
        console=console,
        device=accelerator.device,
        tokenizer=dataset.llm_tokenizer,
        collate_fn_=collate_fn_,
        suffix=f"part{accelerator.process_index}",
    )

    # Wait for all processes to finish
    accelerator.wait_for_everyone()

    # Merge results on main process
    if accelerator.is_main_process:
        merge_results(args, accelerator.num_processes, console)
        console.log("[green]All results merged successfully![/green]")


def merge_results(args, world_size, console, suffix: str = None):
    """Merge results from all GPUs into a single file"""
    console.log("[green]Merging results from all GPUs...[/green]")
    all_results = []
    for rank in range(world_size):
        rank_file = os.path.join(args.gen_dir, f"{args.name}_code_part{rank}.jsonl")
        if os.path.exists(rank_file):
            rank_results = 0
            with open(rank_file, "r") as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        all_results.append(result)
                        rank_results += 1
            console.log(f"[green]Loaded {rank_results} results from GPU {rank}[/green]")
        else:
            console.log(
                f"[yellow]Warning: Results file for GPU {rank} not found: {rank_file}[/yellow]"
            )

    # Write merged results
    final_file = os.path.join(args.gen_dir, f"{args.name}_code.jsonl")
    with open(final_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

    console.log(f"[blue]Merged {len(all_results)} results into {final_file}[/blue]")


if __name__ == "__main__":
    args = parse_args()
    console.log("Generating with Accelerate multi-GPU support")

    print_args(args=args)
    seed_everything(args.seed)

    test_with_accelerate(args=args)

import os
import json
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from itertools import islice
from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
from graph.utils import get_graph
from model.model import get_model
from inference.test import generate_and_save_on_one_dataset
from inference.testcase_generate import testcase_generate
from data.loader import GLMFDataset, collate_fn
from model.model import get_model
import torch.distributed as dist
from functools import partial

# typing
from argparse import Namespace
from rich.console import Console


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get the k-th chunk from a list split into n chunks"""
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


def setup_distributed(rank, world_size, master_port=None):
    """Initialize distributed processing"""
    import random
    import time
    import socket

    os.environ["MASTER_ADDR"] = "localhost"

    # Use provided port or find a free one
    if master_port is None:
        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            master_port = s.getsockname()[1]

    os.environ["MASTER_PORT"] = str(master_port)

    print(f"[GPU {rank}] Setting up distributed processing on port {master_port}")

    try:
        # Add timeout to prevent hanging
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=(
                torch.distributed.default_pg_timeout
                if hasattr(torch.distributed, "default_pg_timeout")
                else None
            ),
        )
        torch.cuda.set_device(rank)
        print(f"[GPU {rank}] Distributed setup successful")
    except Exception as e:
        print(f"[GPU {rank}] Failed to setup distributed processing: {e}")
        print(f"[GPU {rank}] This usually indicates a distributed processing issue")
        raise


def cleanup_distributed():
    """Clean up distributed processing"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Warning: Failed to cleanup distributed process group: {e}")


def eval_model_worker(
    rank: int,
    world_size: int,
    args: Namespace,
):
    """Worker function for each GPU"""
    # Initializing the dataset
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

    console.log(f"Broadcasted args and dataset to all processes.")

    try:
        # Setup distributed processing only for multi-GPU
        if world_size > 1:
            setup_distributed(rank, world_size, args.master_port)

        console.log(f"[green][GPU {rank}] Initializing evaluation worker[/green]")
        console.log(f"[GPU {rank}] Model path: {args.model_weight_path}")

        # Use the actual GPU rank directly instead of modifying CUDA_VISIBLE_DEVICES
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)

        # Get model
        model = get_model(
            args=args,
            tokenizer=dataset.llm_tokenizer,
            rank=rank,
            device=device,
            console=console,
        )

        collate_fn_ = partial(
            collate_fn,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
        )

        # unifying dtype to avoid errors
        for n, p in model.named_parameters():
            if args.dtype == "bf16":
                if p.dtype != torch.bfloat16:
                    p.data = p.data.to(torch.bfloat16)
            elif args.dtype == "fp16":
                if p.dtype != torch.float16:
                    p.data = p.data.to(torch.float16)
        print(f"[GPU {rank}] Model loaded successfully")

        if args.test_on_train:
            data = dict(islice(dataset.train_data.items(), 10))
            keys = list(data.keys())
        else:
            data = dataset.test_data["module"]
            keys = list(data.keys())

        chunk = get_chunk(keys, world_size, rank)
        print(f"[GPU {rank}] Processing {len(keys)} questions")

        if len(chunk) == 0:
            print(f"[GPU {rank}] No questions assigned, exiting")
            return

        # Create output file for this rank
        te_dataset = GLMFDataset(
            data=dataset.test_data["module"],
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
            tokenizer=dataset.llm_tokenizer,
            collate_fn_=collate_fn_,
            suffix=f"part{rank}",
        )

    except Exception as e:
        print(f"[GPU {rank}] Critical error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Only cleanup distributed if we set it up
        if world_size > 1:
            cleanup_distributed()


def merge_results(args, world_size, console: Console, suffix: str = None):
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


def test_on_multiple_gpus(
    args: Namespace,
):
    """Test model using multiple GPUs with distributed processing"""
    world_size = args.num_gpu
    console.log(f"[green]Starting multi-GPU testing on {world_size} GPUs...[/green]")

    if torch.cuda.is_available():
        # Check if distributed training is enabled (this is the case when using Accelerate or torchrun with multi-node)
        if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
            rank = int(os.environ.get("RANK", 0))
            local_rank = int(
                os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())
            )
            world_size = int(os.environ["WORLD_SIZE"])
            device = torch.device("cuda", local_rank)
            console.log(
                f"Distributed training: rank {rank+1}/{world_size}, using device {device}."
            )
            args.num_gpu = world_size
        else:
            # Fallback for single-node training, single or multi GPU.
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                console.log(f"Using {n_gpus} GPUs on a single node.")
                # device = torch.device("cuda:0")
                rank = int(os.environ.get("RANK", 0))
                device = torch.device("cuda", rank)
                args.num_gpu = n_gpus
            else:
                console.log("Using 1 GPU.")
                device = torch.device("cuda:0")
                rank = 0
                local_rank = 0
                args.num_gpu = 1
    else:
        console.log("No GPUs available, using CPU instead.")
        device = torch.device("cpu")
        rank = -1
        local_rank = 0
        args.num_gpu = 0

    try:
        # Use a random port for distributed processing
        import random

        args.master_port = random.randint(10000, 60000)

        mp.spawn(
            eval_model_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )

        # Merge results from all GPUs
        merge_results(args, world_size, console, "module")
    except Exception as e:
        console.log(f"[red]Multi-GPU testing failed[/red]: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    args = parse_args()
    console.log("Generating with multiple GPUs")

    # Initialize the argument parser
    print_args(args=args)
    seed_everything(args.seed)

    test_on_multiple_gpus(args=args)

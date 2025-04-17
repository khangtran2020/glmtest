import os
import torch
import warnings
from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
from graph.utils import get_graph
from train.train import train
import torch.distributed as dist

# typing
from argparse import Namespace
from rich.console import Console

warnings.filterwarnings("ignore")


def main(args: Namespace, logger: Console, device: torch.device, rank: int) -> None:
    console.log("Running on device:", device)

    graph = get_graph(
        args=args,
        graph_type=args.graph_type,
        logger=logger,
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
        data_max_length=args.model_max_length,
        baseline_prompt=args.baseline_prompt,
        debug=args.debug,
        mode=args.mode,
    )
    if dataset is None:
        logger.log("Dataset not found, exiting...")
        return

    if args.mode == "data":
        if args.do_crawl:
            dataset.crawl()
        if args.do_process_raw:
            dataset.process_raw()

    if args.mode == "train":
        if not args.model_debug:
            dataset.prepare_data()
            dataset.train_test_split()

        train(args=args, dataset=dataset, console=console, device=device, rank=rank)


if __name__ == "__main__":
    args = parse_args()
    print_args(args=args)
    seed_everything(args.seed)

    # Device and Distributed Training Setup
    if torch.cuda.is_available():
        # Check if distributed training is enabled (this is the case when using Accelerate or torchrun with multi-node)
        if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
            # Initialize distributed group using the environment variables set by Accelerate
            dist.init_process_group(backend="nccl", init_method="env://")
            rank = int(os.environ.get("RANK", 0))
            # LOCAL_RANK is typically set by Accelerate and indicates the GPU device on the local machine
            local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
            world_size = int(os.environ["WORLD_SIZE"])
            device = torch.device("cuda", local_rank)
            console.log(f"Distributed training: rank {rank}/{world_size}, using device {device}.")
            args.num_gpu = world_size  # Total GPUs across nodes
        else:
            # Fallback for single-node training, single or multi GPU.
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                console.log(f"Using {n_gpus} GPUs on a single node.")
                device = torch.device("cuda:0")
                rank = 0
                args.num_gpu = n_gpus
            else:
                console.log("Using 1 GPU.")
                device = torch.device("cuda:0")
                rank = 0
    else:
        console.log("No GPUs available, using CPU instead.")
        device = torch.device("cpu")
        rank = -1

    main(args=args, logger=console, device=device, rank=rank)

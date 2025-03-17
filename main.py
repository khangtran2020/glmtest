import torch
from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
from graph.utils import get_graph
from train.train import train
from accelerate import Accelerator

# typing
from argparse import Namespace
from rich.console import Console


def main(
    args: Namespace, logger: Console, device: torch.device, accelerator: Accelerator
) -> None:

    # init data

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
    )
    if dataset is None:
        logger.log("Dataset not found, exiting...")
        return

    # data
    if args.mode == "data":
        if args.do_crawl:
            dataset.crawl()
        if args.do_process_raw:
            dataset.process_raw()

    # training
    if args.mode == "train":

        if args.model_debug == False:
            dataset.prepare_data()
            dataset.train_test_split()

        train(
            args=args,
            dataset=dataset,
            console=console,
            device=device,
            accelerator=accelerator,
        )


if __name__ == "__main__":
    args = parse_args()
    print_args(args=args)
    seed_everything(args.seed)

    # count number of gpus
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            console.log(f"Using {n_gpus} GPUs.")
            accelerator = Accelerator()
            device = accelerator.device
            args.num_gpu = n_gpus
        else:
            console.log("Using 1 GPU.")
            device = torch.device("cuda")
            accelerator = None
    else:
        n_gpus = 0
        console.log("No GPUs available, using CPU instead.")
        device = torch.device("cpu")
        accelerator = None

    console.log(f"Using acclerator: {accelerator}")
    main(args=args, logger=console, device=device, accelerator=accelerator)

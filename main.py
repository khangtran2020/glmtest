import torch
from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
from graph.utils import get_graph
from utils.constant import (
    GRAPH_START_TOKEN,
    GRAPH_PAD_TOKEN,
    GRAPH_END_TOKEN,
    FUZZ_START_TOKEN,
    FUZZ_END_TOKEN,
)
from train.train import train

# typing
from argparse import Namespace
from rich.console import Console


def main(args: Namespace, logger: Console, device: torch.device) -> None:

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
        )


if __name__ == "__main__":
    args = parse_args()
    print_args(args=args)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args=args, logger=console, device=device)

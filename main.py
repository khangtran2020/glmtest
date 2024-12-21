from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
from graph.utils import get_graph

# typing
from argparse import Namespace
from rich.console import Console


def main(args: Namespace, logger: Console) -> None:

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
        max_pynguin_run_time=args.max_pynguin_run_time,
        docker_image=args.docker_image,
        num_cpu=args.num_cpu,
        graph=graph,
        debug=args.debug,
    )
    if dataset is None:
        logger.log("Dataset not found, exiting...")
        return

    # data
    if args.mode == "crawl":
        dataset.crawl()
    elif args.mode == "process_raw":
        dataset.process_raw()
    elif args.mode == "test_gen":
        dataset.process_test_gen()


if __name__ == "__main__":
    args = parse_args()
    print_args(args=args)
    seed_everything(args.seed)
    main(args=args, logger=console)

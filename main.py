from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
from graph.joerngraph import JoernGraph

# typing
from argparse import Namespace
from rich.console import Console


def main(args: Namespace, logger: Console) -> None:

    # init data
    graph = JoernGraph(
        host=args.joern_host,
        port=args.joern_port,
        joern_path=args.joern_path,
        graph_path=args.graph_path,
        docker_image=args.joern_docker_image,
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
    )
    if dataset is None:
        logger.log("Dataset not found, exiting...")
        return

    if args.mode == "crawl":
        dataset.crawl()
    elif args.mode == "process":
        dataset.process()
        dataset.graph.init_joern_server()
    elif args.mode == "test_gen":
        dataset.run_test_gen()
    elif args.mode == "test_gen_one":
        for data in dataset.data:
            if data["project"] == args.test_gen_project:
                dataset.run_test_gen_one(data)


if __name__ == "__main__":
    args = parse_args()
    print_args(args=args)
    seed_everything(args.seed)
    main(args=args, logger=console)

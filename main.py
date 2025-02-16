import torch
from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
from data.loader import GLMFDataset, collate_fn
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
        feat_model=args.feat_model,
        llm_model=args.llm_model,
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
    if args.mode == "data":
        if args.do_crawl:
            dataset.crawl()
        if args.do_process_raw:
            dataset.process_raw()

    # training
    if args.mode == "train":
        dataset.prepare_data()
        dataset.train_test_split()

        train_dataset = None
        val_dataset = None
        test_dataset = None
        train_loader = None
        val_loader = None
        test_loader = None

        if args.do_train:
            train_dataset = GLMFDataset(
                data=dataset.train_data,
                tokenizer=dataset.llm_tokenizer,
                max_seq_length=args.max_seq_length,
                debug=args.debug,
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_cpu,
                collate_fn=collate_fn,
            )
        if args.do_val:
            val_dataset = GLMFDataset(
                data=dataset.val_data,
                tokenizer=dataset.llm_tokenizer,
                max_seq_length=args.max_seq_length,
                debug=args.debug,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_cpu,
                collate_fn=collate_fn,
            )
        if args.do_test:
            test_dataset = GLMFDataset(
                data=dataset.test_data,
                tokenizer=dataset.llm_tokenizer,
                max_seq_length=args.max_seq_length,
                debug=args.debug,
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_cpu,
                collate_fn=collate_fn,
            )


if __name__ == "__main__":
    args = parse_args()
    print_args(args=args)
    seed_everything(args.seed)
    main(args=args, logger=console)

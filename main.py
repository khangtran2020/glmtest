import torch
from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
from data.loader import GLMFDataset, collate_fn
from graph.utils import get_graph
from train.train_single_gpu import initialize_trainer_single_gpu
from train.utils import judge_dir
from model.model import GLMFModelForCausalLM, GLMFModelConfig
from train.utils import smart_tokenizer_and_embedding_resize
from utils.constant import (
    GRAPH_START_TOKEN,
    GRAPH_PAD_TOKEN,
    GRAPH_END_TOKEN,
    FUZZ_START_TOKEN,
    FUZZ_END_TOKEN,
)

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
        dataset.prepare_data()
        dataset.train_test_split()

        train_dataset = GLMFDataset(
            data=dataset.train_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
        )
        val_dataset = GLMFDataset(
            data=dataset.val_data,
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
        )

        console.log("Data prepared:")
        console.log(f"Train data: {len(train_dataset)} data points")
        console.log(f"Val data: {len(val_dataset)} data points")

        tokenizer = dataset.llm_tokenizer
        config = GLMFModelConfig(
            llm_model=args.llm_model,
            use_lora=args.use_lora,
            dtype=args.dtype,
            device_map=device,
        )
        model = GLMFModelForCausalLM(config=config)
        special_tokens_dict = {
            {
                "additional_special_tokens": [
                    GRAPH_START_TOKEN,
                    GRAPH_PAD_TOKEN,
                    GRAPH_END_TOKEN,
                    FUZZ_START_TOKEN,
                    FUZZ_END_TOKEN,
                ]
            }
        }
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model.llm_model,
        )
        if args.debug:
            console.log("Model & tokenizer loaded")

        trainer = initialize_trainer_single_gpu(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            args=args,
        )
        if args.debug:
            console.log("Trainer initialized")

        if args.resume_from_checkpoint and judge_dir(args.output_dir):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    print_args(args=args)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args=args, logger=console, device=device)

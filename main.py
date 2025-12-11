import os
import json
import torch
import random
import warnings
from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
from data.core import get_reasoning
from graph.utils import get_graph
from train.train import train
from model.model import get_model
from inference.test import test, eval_bleu_score
from inference.testcase_generate import testcase_generate
from train.utils import load_checkpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from datetime import timedelta
from torch.distributed import init_process_group

# baseline
from baselines.prompt_engineer.run import PromptEngineer
from baselines.codamosa.run import run_codamosa

warnings.filterwarnings("ignore")


def main() -> None:

    args = parse_args()
    console.log("Processing data and initializing model in main process...")

    # Initialize the argument parser
    print_args(args=args)
    seed_everything(args.seed)

    if os.path.exists(args.output_dir) == False:
        os.makedirs(args.output_dir)
    if os.path.exists(args.log_dir) == False:
        os.makedirs(args.log_dir)
    if os.path.exists(args.gen_dir) == False:
        os.makedirs(args.gen_dir)

    if args.model_dir is None:
        args.model_dir = args.output_dir

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
        val_split=100, test_only=True if args.mode == "testgen" else False
    )

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

    # Debugging tokenizer:
    console.log(
        f"[cyan]Tokenizer special tokens:[/cyan]\n{dataset.llm_tokenizer.special_tokens_map}"
    )
    for key, value in dataset.llm_tokenizer.special_tokens_map.items():
        if isinstance(value, str):
            value = dataset.llm_tokenizer.convert_tokens_to_ids(value)
            console.log(f"[cyan]{key}[/cyan]: {value}")
        if isinstance(value, list):
            value = [dataset.llm_tokenizer.convert_tokens_to_ids(v) for v in value]
            console.log(f"[cyan]{key}[/cyan]: {value}")

    model = get_model(
        args=args,
        console=console,
        tokenizer=dataset.llm_tokenizer,
        rank=local_rank,
        device=device,
    )

    if args.mode == "train":

        if args.debug:
            console.log("Model & tokenizer loaded")
            console.log(
                f"Special tokens added to tokenizer and model: {model.config.graph_token_id}"
            )

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-8)

        if args.continue_training:
            model, start_step, optimizer, lr_scheduler = (
                continue_training_from_checkpoint(
                    args=args,
                    model=model,
                    rank=rank,
                    console=console,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
            )
        else:
            start_step = -1

        train(
            args=args,
            dataset=dataset,
            console=console,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            continue_training=args.continue_training,
            start_step=start_step,
            max_num_checkpoint=args.max_num_checkpoint,
            mixed_precision="bf16" if args.dtype == "bf16" else "fp16",
        )

    elif args.mode == "test":

        test(args=args, dataset=dataset, model=model, console=console)

    elif args.mode == "testgen":

        console.log(f"[green]Using device: {device}[/green]")
        testcase_generate(
            args=args,
            dataset=dataset,
            model=model,
            device=device,
            console=console,
            do_generate=args.do_generate,
        )


if __name__ == "__main__":
    main()

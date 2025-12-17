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
from model.model import get_model, extract_metadata_from_graph
from inference.test import test
from inference.testcase_generate import testcase_generate
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
        gnn_mode=args.gnn_mode,
        repo=args.repo,
    )
    if dataset is None:
        console.log("[red]Dataset not found, exiting...[/red]")
        return

    if args.mode == "data":
        dataset.process_raw()
        console.log("Data processing completed. Exiting as mode is 'data'.")
        return

    if not args.baseline_skip_prepare_data:
        if args.mode == "testgen":
            if args.module_path is None:
                dataset.prepare_data_for_test_gen(branch_limit=args.branch_limit)
                for data_n in dataset.processed_data.keys():
                    console.log(
                        f"[blue]Number of test cases for {data_n}: {len(dataset.processed_data[data_n])}[/blue]"
                    )
        if (args.mode == "prepare_reasoning") or (
            (args.mode == "train") and args.train_reasoning
        ):
            dataset.prepare_reasoning_data()
        else:
            dataset.prepare_data()

    if args.mode == "generate_reasoning":
        dataset.filter_by_max_min_tokens(max_tokens=8192, min_tokens=100)
        dataset.sample_for_reasoning(max_samples=10000)
        reasoning_save_path = os.path.join(
            dataset.data_path,
            "reasoning.jsonl",
        )
        if os.path.exists(reasoning_save_path):
            console.log(
                f"[yellow]Reasoning file already exists at {reasoning_save_path}[/yellow]"
            )
            with open(reasoning_save_path, "r") as f:
                generated_reasoning = [json.loads(line) for line in f.readlines()]

            generated_keys = list(
                set([list(obj.keys())[0] for obj in generated_reasoning])
            )
            samples = dataset.processed_data["train"]
            repo_dict = {}
            for key in samples.keys():
                repo = key.split("-")[0]
                if repo not in repo_dict:
                    repo_dict[repo] = []
                repo_dict[repo].append(key)

            removed_keys = []
            for key in generated_keys:
                repo = key.split("-")[0]
                if key in repo_dict[repo]:
                    repo_dict[repo].remove(key)
                    removed_keys.append(key)

            remain_keys = [key for key in generated_keys if key not in removed_keys]
            for key in remain_keys:
                repo = key.split("-")[0]
                chosen_key = random.choice(repo_dict[repo])
                repo_dict[repo].remove(chosen_key)

            samples_to_generate = {}
            for repo in repo_dict:
                for key in repo_dict[repo]:
                    samples_to_generate[key] = samples[key]
        else:
            samples_to_generate = dataset.processed_data["train"]

        console.log(
            f"[green]Generating reasoning for dataset and saving to {reasoning_save_path}[/green]"
        )
        reason_dict = get_reasoning(
            samples=samples_to_generate,
            api_key=args.reason_api_key,
            console=console,
            max_tokens=512,
            model=args.reason_model,
            save_path=reasoning_save_path,
        )
        return  # exit after getting reasoning

    if args.mode == "baseline":
        if args.baseline_prompt_type == "prompt_engineer":
            pe = PromptEngineer(
                args=args,
                model=args.baseline_llm_model,
                api_key=args.baseline_api_key,
                console=console,
            )
            pe.run_prompt_engineering(
                dataset=dataset,
                prompt_type=args.baseline_prompt_type,
                temperature=args.baseline_temp,
                output_path=args.baseline_output_path,
                output_name=args.baseline_output_name,
                max_tokens=args.baseline_max_tokens,
            )
            console.log(
                "Baseline Prompt Engineer completed. Exiting as mode is 'baseline'."
            )
            return
        elif args.baseline_prompt_type == "codamosa":
            if not os.path.exists(os.path.join(dataset.data_path, "test_module.jsonl")):
                raise FileNotFoundError(
                    "test_module.jsonl not found, please crawl the data"
                )
            with open(os.path.join(dataset.data_path, "test_module.jsonl"), "r") as f:
                task_instances = [json.loads(line) for line in f.readlines()]
            run_codamosa(args=args, task_instances=task_instances, console=console)
            return

    if args.mode != "testgen":
        dataset.filter_by_max_min_tokens(
            max_tokens=args.max_seq_length, min_tokens=args.min_seq_length
        )

    if args.repo is not None:
        dataset.prepare_data_by_repo()

    dataset.train_test_split(
        val_split=int(1000), test_only=True if "test" in args.mode else False
    )

    if torch.cuda.is_available():
        # Check if distributed training is enabled (this is the case when using Accelerate or torchrun with multi-node)
        # Also check if DeepSpeed launcher passed --local_rank
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
        elif args.local_rank != -1:
            # DeepSpeed launcher sets --local_rank argument
            local_rank = args.local_rank
            rank = local_rank  # For single-node, local_rank == rank
            device = torch.device("cuda", local_rank)
            # Infer world size from WORLD_SIZE env or default to number of GPUs
            world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
            args.num_gpu = world_size
            console.log(
                f"DeepSpeed distributed: local_rank={local_rank}, world_size={world_size}, device={device}"
            )
        else:
            # Fallback for single-node training, single or multi GPU.
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                console.log(f"Using {n_gpus} GPUs on a single node.")
                # device = torch.device("cuda:0")
                rank = int(os.environ.get("RANK", 0))
                device = torch.device("cuda", rank)
                args.num_gpu = n_gpus
                local_rank = rank
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

    graph_metadata = (
        extract_metadata_from_graph(dataset=dataset)
        if "graph" in args.baseline_prompt
        else None
    )

    if args.mode == "train":
        train(
            args=args,
            dataset=dataset,
            console=console,
            continue_training=args.continue_training,
            max_num_checkpoint=args.max_num_checkpoint,
            metadata=graph_metadata,
            mixed_precision="bf16" if args.dtype == "bf16" else "fp16",
        )
    elif args.mode == "test":
        model = get_model(
            args=args,
            console=console,
            tokenizer=dataset.llm_tokenizer,
            rank=local_rank,
            metadata=graph_metadata,
            device=device,
        )
        test(args=args, dataset=dataset, model=model, console=console)
    elif args.mode == "testgen":
        model = get_model(
            args=args,
            console=console,
            tokenizer=dataset.llm_tokenizer,
            rank=local_rank,
            metadata=graph_metadata,
            device=device,
        )
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

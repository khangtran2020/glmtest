import os
import json
import torch
import warnings
from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
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

    if args.mode == "testgen":
        if args.module_path is None:
            dataset.prepare_data_for_test_gen()
    else:
        dataset.prepare_data()

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
            with open(os.path.join(args.data_path, "test_module.jsonl"), "r") as f:
                task_instances = [json.loads(line) for line in f.readlines()]
            run_codamosa(args=args, task_instances=task_instances, console=console)
            return

    if args.repo is not None:
        dataset.prepare_data_by_repo()

    dataset.train_test_split(
        val_split=int(100), test_only=True if args.mode == "testgen" else False
    )

    console.log(f"Broadcasted args and dataset to all processes.")

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

    if args.mode == "train":
        if args.num_gpu > 1:
            timeout_long_ncll = timedelta(seconds=90000)  # 100 minutes
            init_process_group("nccl", timeout=timeout_long_ncll)

        model = get_model(
            args=args,
            tokenizer=dataset.llm_tokenizer,
            rank=rank,
            device=device,
            console=console,
        )

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
        )
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-8)

        num_params = 0
        for i, param_group in enumerate(optimizer.param_groups):
            group_params = sum(
                p.numel() for p in param_group["params"] if p.requires_grad
            )
            num_params += group_params
            console.log(f"[cyan]Param group {i}: {group_params} parameters[/cyan]")

        console.log(
            f"[yellow]Total parameters tracked by optimizer: {num_params}[/yellow]"
        )

        if args.continue_training:

            if "best_model" in args.checkpoint_path:
                model_path = None
                for file in os.listdir(args.checkpoint_path):
                    if file.endswith(".pt"):
                        model_path = os.path.join(args.checkpoint_path, file)
                        break
                state_dict = torch.load(
                    model_path,
                    map_location=f"cuda:{rank}" if torch.cuda.is_available() else "cpu",
                    weights_only=True,
                )
                # Get current model's state dict
                model_dict = model.state_dict()

                # Filter out mismatched keys or shapes
                filtered_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }

                # Update the model dict
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict)
                start_step = int(model_path.split("step")[-1].split(".pt")[0])
                console.log(
                    f"[cyan]Model weights loaded from {args.checkpoint_path}[/cyan]"
                )
            else:
                assert (
                    args.checkpoint_path is not None
                ), "Checkpoint path must be specified."
                check_point = load_checkpoint(
                    path=args.checkpoint_path, rank=local_rank
                )

                model.load_state_dict(check_point["model_state_dict"])
                optimizer.load_state_dict(check_point["optimizer_state_dict"])
                lr_scheduler.load_state_dict(check_point["scheduler_state_dict"])
                start_step = check_point["global_step"]
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
            mixed_precision="bf16",
        )

    elif args.mode == "test":
        model = get_model(
            args=args,
            tokenizer=dataset.llm_tokenizer,
            rank=rank,
            device=device,
            console=console,
        )
        # unifying dtype to avoid errors
        for n, p in model.named_parameters():
            if args.dtype == "bf16":
                if p.dtype != torch.bfloat16:
                    p.data = p.data.to(torch.bfloat16)
            elif args.dtype == "fp16":
                if p.dtype != torch.float16:
                    p.data = p.data.to(torch.float16)

        console.log(
            f"Model is loaded to device: {model.device} - with type {model.dtype}"
        )
        for name, param in model.named_parameters():
            console.log(f"[yellow]Parameter {name}, dtype: {param.dtype}[/yellow]")

        test(args=args, dataset=dataset, model=model, console=console)

    elif args.mode == "metric":
        assert (
            args.gen_file_path is not None
        ), "File-path must be specified for metric mode."
        eval_bleu_score(args=args, dataset=dataset, console=console)

    elif args.mode == "testgen":

        model = get_model(
            args=args,
            tokenizer=dataset.llm_tokenizer,
            rank=rank,
            device=device,
            console=console,
        )
        # unifying dtype to avoid errors
        if model is not None:
            for n, p in model.named_parameters():
                if args.dtype == "bf16":
                    if p.dtype != torch.bfloat16:
                        p.data = p.data.to(torch.bfloat16)
                    if p.device != device:
                        p.data = p.data.to(device)
                elif args.dtype == "fp16":
                    if p.dtype != torch.float16:
                        p.data = p.data.to(torch.float16)
                    if p.device != device:
                        p.data = p.data.to(device)

            console.log(
                f"Model is loaded to device: {model.device} - with type {model.dtype}"
            )
            for name, param in model.named_parameters():
                console.log(
                    f"[yellow]Parameter {name}, dtype: {param.dtype}, device: {param.device}[/yellow]"
                )

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

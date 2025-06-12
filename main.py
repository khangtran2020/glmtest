import os
import torch
import warnings
from accelerate import Accelerator
from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
from graph.utils import get_graph
from train.train import train, GLMFModelForCausalLM, GLMFModelConfig
from train.test import test, testCache, getMetric
from accelerate.utils import broadcast_object_list
from train.utils import load_checkpoint
from utils.constant import (
    GRAPH_START_TOKEN,
    GRAPH_PAD_TOKEN,
    GRAPH_END_TOKEN,
)

from transformers import get_scheduler
from torch.optim import AdamW
from utils.utils import log_ram_usage

# typing
from argparse import Namespace
from rich.console import Console
from datetime import timedelta
from torch.distributed import init_process_group

warnings.filterwarnings("ignore")


def main() -> None:

    args = parse_args()
    if args.num_gpu > 1:
        timeout_long_ncll = timedelta(seconds=90000)  # 100 minutes
        init_process_group("nccl", timeout=timeout_long_ncll)
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     mixed_precision=args.dtype,
    #     log_with="wandb",
    #     project_dir=args.log_dir,
    # )

    # Initialize args, logger, model and dataset:
    # if accelerator.is_main_process:
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

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        args.num_gpu = int(os.environ["WORLD_SIZE"])
    else:
        args.num_gpu = torch.cuda.device_count()

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
        model_name=args.model_name,
        data_max_length=args.max_seq_length,
        baseline_prompt=args.baseline_prompt,
        debug=args.debug,
        mode=args.mode,
        graph_sampling=args.graph_sampling,
        n_hops=args.n_layers,
        max_tokens=args.max_seq_length,
    )
    if dataset is None:
        console.log("Dataset not found, exiting...")
        return

    ram_usage = log_ram_usage()
    console.log(f"Dataset loaded - RAM usage: {ram_usage:.2f} MB")

    if args.mode == "data":
        if args.do_crawl:
            dataset.crawl()
        if args.do_process_raw:
            dataset.process_raw()
        return

    if not args.model_debug:
        dataset.prepare_data()
        dataset.train_test_split(val_split=1000, test_split=200)

    # else:
    #     dataset = [None]
    #     args = [None]

    # broadcast the args and dataset to all processes
    # dataset = accelerator.send
    # # # (dataset, src_rank=0)
    # dataset = broadcast_object_list(object_list=dataset, from_process=0)
    # args = broadcast_object_list(object_list=args, from_process=0)
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
                f"Distributed training: rank {rank}/{world_size}, using device {device}."
            )
        else:
            # Fallback for single-node training, single or multi GPU.
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                console.log(f"Using {n_gpus} GPUs on a single node.")
                # device = torch.device("cuda:0")
                rank = int(os.environ.get("RANK", 0))
                device = torch.device("cuda", rank)
            else:
                console.log("Using 1 GPU.")
                device = torch.device("cuda:0")
                rank = 0
    else:
        console.log("No GPUs available, using CPU instead.")
        device = torch.device("cpu")
        rank = -1

    config = GLMFModelConfig(
        llm_model=args.llm_model,
        use_lora=args.use_lora,
        dtype=args.dtype,
        mode=args.gnn_mode,
        in_feats=args.in_feats,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        num_head=args.num_head,
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    # if config.model_type not in ["llama", "qwen2"]:
    #     raise ValueError(
    #         f"Model type {config.model_type} is not supported. Please use 'llama' or 'qwen2'."
    #     )

    if args.mode == "train":

        model = GLMFModelForCausalLM(
            config=config,
            tokenizer=dataset.llm_tokenizer,
            baseline_prompt=args.baseline_prompt,
            multi_gpu=True if args.num_gpu > 1 else False,
            debug=args.debug,
            rank=rank,
            training=True,
        )
        model.llm_model.gradient_checkpointing_enable()
        model.config.graph_token_id = [
            dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
            dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
            dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
        ]
        if args.model_debug:
            return

        if args.debug:
            console.log("Model & tokenizer loaded")
            console.log(
                f"Special tokens added to tokenizer and model: {model.config.graph_token_id}"
            )

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
        )
        lr_scheduler = get_scheduler(
            name="cosine_with_restarts",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=args.num_train_epochs,
        )

        if args.continue_training:
            assert (
                args.checkpoint_path is not None
            ), "Checkpoint path must be specified."
            check_point = load_checkpoint(path=args.checkpoint_path)

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
            mixed_precision="bf16",
        )

        if args.do_test:

            model_path = os.path.join(args.output_dir, args.name)
            model_path = os.path.join(model_path, "final_model")
            model.load_state_dict(
                torch.load(os.path.join(model_path, "model_weight.pt"))
            )
            # config = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda")
            # console.log(f"Config loaded from: {model_path}\n {config}")
            # config.vocab_size = dataset.llm_tokenizer.vocab_size
            # console.log("Config vocab size:", config.vocab_size)
            # model = GLMFModelForCausalLM.from_pretrained(
            #     pretrained_model_name_or_path=model_path, device_map="cpu"
            # )
            console.log(f"Model is loaded to device: {model.device}")
            test(args=args, dataset=dataset, model=model, console=console)

    elif args.mode == "test":
        # load model
        assert (
            args.model_weight_path is not None
        ), "Model directory must be specified for testing."
        # config = GLMFModelConfig.from_pretrained(model_path, device_map="auto")
        # config.vocab_size = dataset.llm_tokenizer.vocab_size
        # model = GLMFModelForCausalLM.from_pretrained(model_path, config=config)
        # model = GLMFModelForCausalLM.from_pretrained(args.model_weight_path)
        # model = GLMFModelForCausalLM.from_pretrained(
        #     pretrained_model_name_or_path=args.model_weight_path, device_map="cpu"
        # )
        config = GLMFModelConfig(
            llm_model=args.llm_model,
            use_lora=False,
            dtype=args.dtype,
            mode=args.gnn_mode,
            in_feats=args.in_feats,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            num_head=args.num_head,
            dropout=args.dropout,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            device_map="cuda",
        )

        # if config.model_type not in ["llama", "qwen2", "qwen3"]:
        #     raise ValueError(
        #         f"Model type {config.model_type} is not supported. Please use 'llama' or 'qwen2', 'qwen3'."
        #     )

        model = GLMFModelForCausalLM(
            config=config,
            tokenizer=dataset.llm_tokenizer,
            baseline_prompt=args.baseline_prompt,
            debug=args.debug,
            rank=rank,
            training=False,
        )

        model.config.graph_token_id = [
            dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
            dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
            dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
        ]

        model.load_state_dict(
            torch.load(os.path.join(args.model_weight_path, "model_weight.pt"))
        )
        console.log(f"Model is loaded to device: {model.device}")
        test(args=args, dataset=dataset, model=model, console=console)

    elif args.mode == "metric":

        assert (
            args.gen_file_path is not None
        ), "File-path must be specified for metric mode."

        getMetric(args=args, dataset=dataset, console=console)


if __name__ == "__main__":
    main()

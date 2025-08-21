import os
import torch
import warnings
from config import parse_args
from utils.console import console
from utils.utils import print_args, seed_everything
from data.utils import get_dataset
from graph.utils import get_graph
from train.train import train
from model.model import GLMFModelForCausalLM, GLMFModelConfig, GLMFModelFuzzing
from train.test import test, eval_bleu_score
from train.utils import load_checkpoint
from utils.constant import (
    GRAPH_START_TOKEN,
    GRAPH_PAD_TOKEN,
    GRAPH_END_TOKEN,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from utils.utils import log_ram_usage
from datetime import timedelta
from torch.distributed import init_process_group

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
        model_name=args.model_name,
        data_max_length=args.max_seq_length,
        baseline_prompt=args.baseline_prompt,
        debug=args.debug,
        mode=args.mode,
        graph_sampling=args.graph_sampling,
        n_hops=args.n_layers,
        max_tokens=args.max_seq_length,
        raw_overwrite=args.raw_overwrite,
    )
    if dataset is None:
        console.log("Dataset not found, exiting...")
        return

    if args.debug:
        ram_usage = log_ram_usage()
        console.log(f"Dataset loaded - RAM usage: {ram_usage:.2f} MB")

    if args.mode == "data":
        if args.do_crawl:
            dataset.crawl()
        if args.do_process_raw:
            dataset.process_raw()
        return

    dataset.prepare_data()
    dataset.train_test_split(val_split=1000, test_split=200)

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

    if args.num_gpu > 1:
        timeout_long_ncll = timedelta(seconds=90000)  # 100 minutes
        init_process_group("nccl", timeout=timeout_long_ncll)

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

        if args.fuzz_model:

            # Logging the vocabulary size of the tokenizer
            console.log(f"[cyan]Training with fuzzing mode:[/cyan]")

            glmf_model_config = GLMFModelConfig(
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
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )

            layer_indices = [
                i for i in range(args.start_fuzz_layer_index, args.end_fuzz_layer_index)
            ]
            glmf_model = GLMFModelForCausalLM(
                config=glmf_model_config,
                tokenizer=dataset.llm_tokenizer,
                baseline_prompt=args.baseline_prompt,
                debug=args.debug,
                rank=rank,
                multi_gpu=True if args.num_gpu > 1 else False,
                is_training=True,
            )

            glmf_model.config.graph_token_id = [
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
            ]

            if args.model_weight_path is not None:
                for file in os.listdir(args.model_weight_path):
                    if file.endswith(".pt"):
                        state_dict = torch.load(
                            os.path.join(args.model_weight_path, file),
                            map_location=f"cuda:{rank}" if args.num_gpu > 1 else "cpu",
                        )
                        glmf_model.load_state_dict(state_dict)

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

            config.graph_token_id = [
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
            ]

            console.log(
                f"Config vocab size: {config.vocab_size}, "
                f"Tokenizer vocab size: {dataset.llm_tokenizer.vocab_size}"
            )

            model = GLMFModelFuzzing(
                config=config,
                rank=local_rank,
                tokenizer=dataset.llm_tokenizer,
                baseline_prompt=args.baseline_prompt,
                multi_gpu=True if args.num_gpu > 1 else False,
                debug=args.debug,
                is_training=True,
                layer_indices=layer_indices,
                glmf_model=glmf_model,
                glmf_model_weight_path=args.model_weight_path,
                kl_g_reg=args.kl_g_reg,
                kl_d_reg=args.kl_d_reg,
            )

        else:
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

            model = GLMFModelForCausalLM(
                config=config,
                tokenizer=dataset.llm_tokenizer,
                baseline_prompt=args.baseline_prompt,
                multi_gpu=True if args.num_gpu > 1 else False,
                debug=args.debug,
                rank=local_rank,
                is_training=True,
            )

            model.llm_model.gradient_checkpointing_enable()
            model.config.graph_token_id = [
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
            ]

            console.log(
                f"Attention implementation of the model is: {model.llm_model.config._attn_implementation}"
            )

        if args.debug:
            console.log("Model & tokenizer loaded")
            console.log(
                f"Special tokens added to tokenizer and model: {model.config.graph_token_id}"
            )

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
        )
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-8)

        if args.continue_training:
            assert (
                args.checkpoint_path is not None
            ), "Checkpoint path must be specified."
            check_point = load_checkpoint(path=args.checkpoint_path, rank=local_rank)

            model.load_state_dict(check_point["model_state_dict"])
            optimizer.load_state_dict(check_point["optimizer_state_dict"])
            lr_scheduler.load_state_dict(check_point["scheduler_state_dict"])
            start_step = check_point["global_step"]
            if args.debug:
                console.log(
                    f"Checkpoint loaded from {args.checkpoint_path}, starting from step {start_step}."
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
            mixed_precision="bf16",
        )

    elif args.mode == "test":
        # load model
        assert (
            args.model_weight_path is not None
        ), "Model directory must be specified for testing."

        if "current_checkpoint" in args.model_weight_path:
            use_lora = True
        else:
            use_lora = False

        if args.fuzz_model:

            # Logging the vocabulary size of the tokenizer
            console.log(f"[cyan]Testing with fuzzing mode:[/cyan]")

            glmf_model_config = GLMFModelConfig(
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
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )

            layer_indices = [
                i for i in range(args.start_fuzz_layer_index, args.end_fuzz_layer_index)
            ]
            glmf_model = GLMFModelForCausalLM(
                config=glmf_model_config,
                tokenizer=dataset.llm_tokenizer,
                baseline_prompt=args.baseline_prompt,
                debug=args.debug,
                rank=rank,
                multi_gpu=True if args.num_gpu > 1 else False,
                is_training=False,
            )

            glmf_model.config.graph_token_id = [
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
            ]

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
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )

            config.graph_token_id = [
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
            ]

            model = GLMFModelFuzzing(
                config=config,
                rank=local_rank,
                tokenizer=dataset.llm_tokenizer,
                baseline_prompt=args.baseline_prompt,
                multi_gpu=True if args.num_gpu > 1 else False,
                debug=args.debug,
                is_training=False,
                layer_indices=layer_indices,
                glmf_model=glmf_model,
                glmf_model_weight_path=args.model_weight_path,
                kl_g_reg=args.kl_g_reg,
                kl_d_reg=args.kl_d_reg,
            )
        else:
            config = GLMFModelConfig(
                llm_model=args.llm_model,
                use_lora=use_lora,
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

            model = GLMFModelForCausalLM(
                config=config,
                tokenizer=dataset.llm_tokenizer,
                baseline_prompt=args.baseline_prompt,
                debug=args.debug,
                rank=rank,
                multi_gpu=True if args.num_gpu > 1 else False,
                is_training=False,
            )

            model.config.graph_token_id = [
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
                dataset.llm_tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
            ]

        # take .pt file from the model_weight_path
        for file in os.listdir(args.model_weight_path):
            if file.endswith(".pt"):
                state_dict = torch.load(
                    os.path.join(args.model_weight_path, file),
                    map_location=f"cuda:{rank}" if args.num_gpu > 1 else "cpu",
                )
                model.load_state_dict(state_dict)
                if use_lora:
                    model.llm_model = model.llm_model.merge_and_unload()

        # model = model.to(dtype=torch.float)
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


if __name__ == "__main__":
    main()

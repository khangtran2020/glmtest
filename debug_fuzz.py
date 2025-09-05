import os
import torch
import warnings
from copy import deepcopy
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
from model.gnn import GRAPH_KEYS
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from utils.utils import log_ram_usage
from datetime import timedelta
from torch.distributed import init_process_group
from data.loader import GLMFDataset, collate_fn
from accelerate import Accelerator
from torch.utils.data import DataLoader

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

    glmf_model.llm_model = deepcopy(model.llm_model)
    glmf_model.config.use_lora = True

    total_params_glmf_model = 0
    for name, param in glmf_model.named_parameters():
        if param.requires_grad:
            console.log(
                f"[blue]{name}: {param.numel()} parameters, shape={tuple(param.shape)}[/blue]"
            )
            total_params_glmf_model += param.numel()

    total_params_model = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            console.log(
                f"[blue]{name}: {param.numel()} parameters, shape={tuple(param.shape)}[/blue]"
            )
            total_params_model += param.numel()

    console.log(
        f"[green]Total trainable parameters from glmf_model: {total_params_glmf_model}[/green]"
    )
    console.log(
        f"[cyan]Total trainable parameters from model: {total_params_model}[/cyan]"
    )

    assert total_params_model == total_params_glmf_model, "Parameter count mismatch!"

    optimizer_model = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )
    lr_scheduler_model = CosineAnnealingLR(optimizer_model, T_max=100, eta_min=5e-8)

    optimizer_glmf_model = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )
    lr_scheduler_glmf_model = CosineAnnealingLR(
        optimizer_glmf_model, T_max=100, eta_min=5e-8
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb",
        project_dir=args.log_dir,
    )

    # Initialize W&B run if main process
    accelerator.init_trackers(
        project_name="GLMFuzz",
        config={
            "model_name": args.llm_model,
            "dataset": args.data,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size
            * args.gradient_accumulation_steps
            * accelerator.num_processes,
            "max_steps": args.num_train_epochs,
            "mixed_precision": "bf16",
            "seed": args.seed,
        },
        init_kwargs={"wandb": {"name": args.name}},
    )

    tokenizer = dataset.llm_tokenizer
    device = accelerator.device
    accelerator.print(f"Using {accelerator.num_processes} devices")
    accelerator.print(f"Mixed precision: bf16")

    tr_dataset = GLMFDataset(
        data=dataset.train_data,
        tokenizer=dataset.llm_tokenizer,
        max_seq_length=args.max_seq_length,
        debug=args.debug,
        n_hops=dataset.n_hops,
        dtype=args.dtype,
        num_gpus=args.num_gpu,
        logger=console,
    )
    va_dataset = GLMFDataset(
        data=dataset.val_data,
        tokenizer=dataset.llm_tokenizer,
        max_seq_length=args.max_seq_length,
        debug=args.debug,
        n_hops=dataset.n_hops,
        dtype=args.dtype,
        num_gpus=args.num_gpu,
        logger=console,
    )
    tr_loader = DataLoader(
        tr_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    va_loader = DataLoader(
        va_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    device = accelerator.device
    config = model.config
    model, optimizer_model, lr_scheduler_model = accelerator.prepare(
        model, optimizer_model, lr_scheduler_model
    )
    glmf_model, optimizer_glmf_model, lr_scheduler_glmf_model = accelerator.prepare(
        glmf_model, optimizer_glmf_model, lr_scheduler_glmf_model
    )
    global_step = 0
    previous_checkpoint_step = -1
    best_val_loss = 10000.0

    glmf_model.train()
    model.train()

    for step, batch in enumerate(tr_loader):

        batch_size = batch["input"]["input_ids"].size(0)

        for i in range(batch_size):

            batch_input = batch["input"].copy()
            if "token_type_ids" in batch_input:
                batch_input.pop("token_type_ids")

            micro_input = {
                "input_ids": batch_input["input_ids"][i].to(device),
                "attention_mask": batch_input["attention_mask"][i].to(device),
                "labels": batch_input["labels"][i].to(device),
            }

            if "graph" in args.baseline_prompt:
                graph = batch["graph"][i]
                for key in GRAPH_KEYS:
                    if key in graph.keys():
                        graph[key] = graph[key].to(device)

                graph_mask = batch["graph_mask"][i].to(device)
                graph_token_index = torch.where(
                    micro_input["input_ids"] == model.config.graph_token_id[1]
                )[1].tolist()
            else:
                graph = None
                graph_mask = None
                graph_token_index = None

            with accelerator.accumulate(model):
                outputs_model = model(
                    **micro_input,
                    step=global_step,
                    graph=graph,
                    graph_mask=graph_mask,
                    graph_token_index=graph_token_index,
                )

                outputs_glmf_model = glmf_model(
                    **micro_input,
                    step=global_step,
                    graph=graph,
                    graph_mask=graph_mask,
                    graph_token_index=graph_token_index,
                )

                loss_model = outputs_model.loss
                loss_glmf_model = outputs_glmf_model.loss

                logits_model = outputs_model.logits
                logits_glmf_model = outputs_glmf_model.logits

                console.log(
                    f"[blue]Step {step}, Micro-batch {i+1}/{batch_size}[/blue], [green]Loss model: {loss_model.item():.4f}[/green], [yellow]Loss glmf_model: {loss_glmf_model.item():.4f}[yellow]"
                )
                console.log(
                    f"[blue]Step {step}, Micro-batch {i+1}/{batch_size}[/blue], [green]Logits model: {logits_model[:, -1, :5]}[/green], [yellow]Logits glmf_model: {logits_glmf_model[:, -1, :5]}[yellow]"
                )

                accelerator.backward(loss_model)
                accelerator.backward(loss_glmf_model)

                # Check gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if torch.isnan(param.grad).any():
                            console.log(
                                f"[red]NaN gradient detected in model parameter {name} at step {global_step}[/red]"
                            )
                        console.log(
                            f"[green] At step {global_step}, model parameter {name} has grad norm {grad_norm:.4f}[/green]"
                        )

                # Check gradients
                for name, param in glmf_model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if torch.isnan(param.grad).any():
                            console.log(
                                f"[red]NaN gradient detected in glmf_model parameter {name} at step {global_step}[/red]"
                            )
                        console.log(
                            f"[green] At step {global_step}, glmf_model parameter {name} has grad norm {grad_norm:.4f}[/green]"
                        )

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            accelerator.clip_grad_norm_(glmf_model.parameters(), 1.0)
            optimizer_model.step()
            lr_scheduler_model.step()
            optimizer_model.zero_grad()

            optimizer_glmf_model.step()
            lr_scheduler_glmf_model.step()
            optimizer_glmf_model.zero_grad()

            # Check model params' values after update
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_norm = param.norm().item()
                    if torch.isnan(param).any():
                        console.log(
                            f"[red]NaN value detected in model parameter {name} at step {global_step}[/red]"
                        )
                    console.log(
                        f"[green] At step {global_step}, model parameter {name} has norm {param_norm:.4f}[/green]"
                    )
            for name, param in glmf_model.named_parameters():
                if param.requires_grad:
                    param_norm = param.norm().item()
                    if torch.isnan(param).any():
                        console.log(
                            f"[red]NaN value detected in glmf_model parameter {name} at step {global_step}[/red]"
                        )
                    console.log(
                        f"[green] At step {global_step}, glmf_model parameter {name} has norm {param_norm:.4f}[/green]"
                    )
            break

    # Check save model process
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_glmf_model = accelerator.unwrap_model(glmf_model)

    if unwrapped_model.config.use_lora == True:
        unwrapped_model.llm_model = unwrapped_model.llm_model.merge_and_unload()
        unwrapped_glmf_model.llm_model = (
            unwrapped_glmf_model.llm_model.merge_and_unload()
        )

    # Logs model and glmf_model parameter norms before saving
    for name, param in unwrapped_model.named_parameters():
        if param.requires_grad:
            param_norm = param.norm().item()
            console.log(
                f"[green] Before saving, model parameter {name} has norm {param_norm:.4f}[/green]"
            )
    for name, param in unwrapped_glmf_model.named_parameters():
        if param.requires_grad:
            param_norm = param.norm().item()
            console.log(
                f"[blue] Before saving, glmf_model parameter {name} has norm {param_norm:.4f}[/blue]"
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()

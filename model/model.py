import os
import json
import torch
from model.glmf import GLMFModelConfig, GLMFModelForCausalLM
from model.glmffuzz import GLMFModelFuzzing
from data.core import Data
from rich.console import Console
from argparse import Namespace
from transformers import PreTrainedTokenizer
from train.utils import load_checkpoint
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import (
    BaseStorage,
    GlobalStorage,
    NodeStorage,
    EdgeStorage,
)
from utils.constant import (
    GRAPH_START_TOKEN,
    GRAPH_PAD_TOKEN,
    GRAPH_END_TOKEN,
)


def get_model(
    args: Namespace,
    console: Console,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    device: torch.device,
    metadata: tuple = None,
    use_zero3: bool = False,
):
    if args.mode == "train":
        model = get_model_train(
            args=args,
            console=console,
            tokenizer=tokenizer,
            rank=rank,
            metadata=metadata,
            use_zero3=use_zero3,
        )
    elif args.mode == "test":
        model = get_model_test(
            args=args,
            console=console,
            tokenizer=tokenizer,
            rank=rank,
            metadata=metadata,
        )
    elif args.mode == "testgen":
        model = get_model_testgen(
            args=args,
            console=console,
            rank=rank,
            tokenizer=tokenizer,
            metadata=metadata,
            device=device,
        )
    else:
        console.log(f"Mode {args.mode} not using model.")
        return None
    # for name, param in model.named_parameters():
    #     console.log(f"[yellow]Parameter {name}, dtype: {param.dtype}[/yellow]")
    return model


def get_model_train(
    args: Namespace,
    console: Console,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    metadata: tuple = None,
    use_zero3: bool = False,
):
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
            tokenizer=tokenizer,
            baseline_prompt=args.baseline_prompt,
            debug=args.debug,
            rank=rank,
            multi_gpu=True if args.num_gpu > 1 else False,
            is_training=True,
        )

        # set metadata as property of glmf_model
        glmf_model.metadata = metadata
        if metadata is not None:
            glmf_model.gnn = to_hetero(glmf_model.gnn, metadata=metadata, aggr="sum")
            console.log(
                f"[blue]Converted GNN to heterogeneous with metadata: {metadata}[/blue]"
            )

        glmf_model.config.graph_token_id = [
            tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
        ]

        if args.model_weight_path is not None:
            for file in os.listdir(args.model_weight_path):
                if file.endswith(".pt"):
                    state_dict = torch.load(
                        os.path.join(args.model_weight_path, file),
                        map_location=f"cuda:{rank}" if args.num_gpu > 1 else "cpu",
                        weights_only=True,
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
            tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
        ]

        console.log(
            f"Config vocab size: {config.vocab_size}, "
            f"Tokenizer vocab size: {tokenizer.vocab_size}"
        )

        model = GLMFModelFuzzing(
            config=config,
            rank=rank,
            tokenizer=tokenizer,
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

        if args.only_nvib:

            # freeze all params first
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze NVIB params
            for name, param in model.named_parameters():
                if "nvib_layer" in name:
                    param.requires_grad = True
                    console.log(f"Parameter {name} is set to be trainable.")
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
            device_map=(
                None if use_zero3 else ("cuda" if torch.cuda.is_available() else "cpu")
            ),
        )
        model = GLMFModelForCausalLM(
            config=config,
            tokenizer=tokenizer,
            baseline_prompt=args.baseline_prompt,
            multi_gpu=True if args.num_gpu > 1 else False,
            debug=args.debug,
            rank=rank,
            is_training=True,
            use_zero3=use_zero3,
        )
        model.metadata = metadata
        if metadata is not None:
            model.gnn = to_hetero(model.gnn, metadata=metadata, aggr="sum")
            console.log(
                f"[blue]Converted GNN to heterogeneous with metadata: {metadata}[/blue]"
            )
        if args.model_weight_path is not None:
            if args.model_weight_path.endswith(".pt"):
                weights_path = args.model_weight_path
            else:
                console.log(f"[red]Finding weights from {args.model_weight_path}[/red]")
                for file in os.listdir(args.model_weight_path):
                    if file.endswith(".pt"):
                        weights_path = os.path.join(args.model_weight_path, file)
                        break
            console.log(f"[red]Loading weight from {weights_path}[/red]")
            check_point = load_checkpoint(path=weights_path, rank=rank)
            state_dict = check_point["model_state_dict"]
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            if missing_keys:
                console.log(
                    f"[yellow]Missing keys in checkpoint: {len(missing_keys)} keys[/yellow]"
                )
            if unexpected_keys:
                console.log(
                    f"[yellow]Unexpected keys in checkpoint (ignored): {len(unexpected_keys)} keys[/yellow]"
                )
                # Log sample of unexpected keys for debugging
                sample_unexpected = list(unexpected_keys)[:3]
                console.log(
                    f"[yellow]Sample unexpected keys: {sample_unexpected}...[/yellow]"
                )

        model.llm_model.gradient_checkpointing_enable()
        model.config.graph_token_id = [
            tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
        ]

        console.log(
            f"Attention implementation of the model is: {model.llm_model.config._attn_implementation}"
        )

    # for n, p in model.named_parameters():
    #     if args.dtype == "bf16":
    #         if p.dtype != torch.bfloat16:
    #             p.data = p.data.to(torch.bfloat16)
    #     elif args.dtype == "fp16":
    #         if p.dtype != torch.float16:
    #             p.data = p.data.to(torch.float16)

    return model


def get_model_test(
    args: Namespace,
    console: Console,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    metadata: tuple = None,
):
    # load model

    if rank == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{rank}")

    assert (
        args.model_weight_path is not None
    ), "Model directory must be specified for testing."

    use_lora = True

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
            tokenizer=tokenizer,
            baseline_prompt=args.baseline_prompt,
            debug=args.debug,
            rank=rank,
            multi_gpu=True if args.num_gpu > 1 else False,
            is_training=False,
        )
        glmf_model.metadata = metadata
        if metadata is not None:
            glmf_model.gnn = to_hetero(glmf_model.gnn, metadata=metadata, aggr="sum")
            console.log(
                f"[blue]Converted GNN to heterogeneous with metadata: {metadata}[/blue]"
            )

        glmf_model.config.graph_token_id = [
            tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
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
            tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
        ]

        model = GLMFModelFuzzing(
            config=config,
            rank=rank,
            tokenizer=tokenizer,
            baseline_prompt=args.baseline_prompt,
            multi_gpu=True if args.num_gpu > 1 else False,
            debug=args.debug,
            is_training=False,
            layer_indices=layer_indices,
            glmf_model=glmf_model,
            glmf_model_weight_path=args.model_weight_path,
            kl_g_reg=args.kl_g_reg,
            kl_d_reg=args.kl_d_reg,
            fuzzing=args.fuzzing,
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
            tokenizer=tokenizer,
            baseline_prompt=args.baseline_prompt,
            debug=args.debug,
            rank=rank,
            multi_gpu=True if args.num_gpu > 1 else False,
            is_training=False,
        )

        model.metadata = metadata
        if metadata is not None:
            model.gnn = to_hetero(model.gnn, metadata=metadata, aggr="sum")
            console.log(
                f"[blue]Converted GNN to heterogeneous with metadata: {metadata}[/blue]"
            )

        model.config.graph_token_id = [
            tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
        ]

    if args.model_weight_path is not None:
        if args.model_weight_path.endswith(".pt"):
            weights_path = args.model_weight_path
        else:
            console.log(f"[red]Finding weights from {args.model_weight_path}[/red]")
            for file in os.listdir(args.model_weight_path):
                if file.endswith(".pt"):
                    weights_path = os.path.join(args.model_weight_path, file)
                    break
        console.log(f"[red]Loading weight from {weights_path}[/red]")
        check_point = load_checkpoint(path=weights_path, rank=rank)
        state_dict = check_point["model_state_dict"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            console.log(
                f"[yellow]Missing keys in checkpoint: {len(missing_keys)} keys[/yellow]"
            )
        if unexpected_keys:
            console.log(
                f"[yellow]Unexpected keys in checkpoint (ignored): {len(unexpected_keys)} keys[/yellow]"
            )
            # Log sample of unexpected keys for debugging
            sample_unexpected = list(unexpected_keys)[:3]
            console.log(
                f"[yellow]Sample unexpected keys: {sample_unexpected}...[/yellow]"
            )

    if use_lora:
        model.llm_model = model.llm_model.merge_and_unload()
    console.log(f"[red]Model weights loaded from {args.model_weight_path}[/red]")
    for n, p in model.named_parameters():
        if args.dtype == "bf16":
            p.data = p.data.to(torch.bfloat16)
            p.data = p.data.to(device)
        elif args.dtype == "fp16":
            p.data = p.data.to(torch.float16)
            p.data = p.data.to(device)

    console.log(f"Model is loaded to device: {model.device} - with type {model.dtype}")
    return model


def get_model_testgen(
    args: Namespace,
    console: Console,
    rank: int,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    metadata: tuple = None,
):
    # load model
    assert (
        args.model_weight_path is not None
    ), "Model directory must be specified for testing."
    use_lora = True

    model = None
    if args.do_generate:
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
                tokenizer=tokenizer,
                baseline_prompt=args.baseline_prompt,
                debug=args.debug,
                rank=rank,
                multi_gpu=True if args.num_gpu > 1 else False,
                is_training=False,
            )
            glmf_model.metadata = metadata
            if metadata is not None:
                glmf_model.gnn = to_hetero(
                    glmf_model.gnn, metadata=metadata, aggr="sum"
                )
                console.log(
                    f"[blue]Converted GNN to heterogeneous with metadata: {metadata}[/blue]"
                )

            glmf_model.config.graph_token_id = [
                tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
                tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
                tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
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
                tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
                tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
                tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
            ]

            model = GLMFModelFuzzing(
                config=config,
                rank=rank,
                tokenizer=tokenizer,
                baseline_prompt=args.baseline_prompt,
                multi_gpu=True if args.num_gpu > 1 else False,
                debug=args.debug,
                is_training=False,
                layer_indices=layer_indices,
                glmf_model=glmf_model,
                glmf_model_weight_path=args.model_weight_path,
                kl_g_reg=args.kl_g_reg,
                kl_d_reg=args.kl_d_reg,
                fuzzing=args.fuzzing,
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
                tokenizer=tokenizer,
                baseline_prompt=args.baseline_prompt,
                debug=args.debug,
                rank=rank,
                multi_gpu=True if args.num_gpu > 1 else False,
                is_training=False,
            )

            model.metadata = metadata
            if metadata is not None:
                model.gnn = to_hetero(model.gnn, metadata=metadata, aggr="sum")
                console.log(
                    f"[blue]Converted GNN to heterogeneous with metadata: {metadata}[/blue]"
                )

            model.config.graph_token_id = [
                tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
                tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
                tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
            ]
        console.log(f"[red]Finding weights from {args.model_weight_path}[/red]")

        if args.model_weight_path is not None:
            if args.model_weight_path.endswith(".pt"):
                weights_path = args.model_weight_path
            else:
                console.log(f"[red]Finding weights from {args.model_weight_path}[/red]")
                for file in os.listdir(args.model_weight_path):
                    if file.endswith(".pt"):
                        weights_path = os.path.join(args.model_weight_path, file)
                        break
            console.log(f"[red]Loading weight from {weights_path}[/red]")
            check_point = load_checkpoint(path=weights_path, rank=rank)
            state_dict = check_point["model_state_dict"]
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            if missing_keys:
                console.log(
                    f"[yellow]Missing keys in checkpoint: {len(missing_keys)} keys[/yellow]"
                )
            if unexpected_keys:
                console.log(
                    f"[yellow]Unexpected keys in checkpoint (ignored): {len(unexpected_keys)} keys[/yellow]"
                )
                # Log sample of unexpected keys for debugging
                sample_unexpected = list(unexpected_keys)[:3]
                console.log(
                    f"[yellow]Sample unexpected keys: {sample_unexpected}...[/yellow]"
                )
        if use_lora:
            model.llm_model = model.llm_model.merge_and_unload()
        console.log(f"[red]Model weights loaded from {args.model_weight_path}[/red]")

        # model = model.to(dtype=torch.float)
        # unifying dtype to avoid errors
        for n, p in model.named_parameters():
            if args.dtype == "bf16":
                p.data = p.data.to(torch.bfloat16)
                p.data = p.data.to(device)
            elif args.dtype == "fp16":
                p.data = p.data.to(torch.float16)
                p.data = p.data.to(device)
        console.log(
            f"Model is loaded to device: {model.device} - with type {model.dtype}"
        )
        # for name, param in model.named_parameters():
        #     console.log(
        #         f"[yellow]Parameter {name}, dtype: {param.dtype}, device: {param.device}[/yellow]"
        #     )
    return model


def continue_training_from_checkpoint(
    args: Namespace,
    model: GLMFModelForCausalLM,
    local_rank: int,
    console: Console,
):
    if args.continue_training:

        # assert args.checkpoint_path is not None, "Checkpoint path must be specified."
        if args.checkpoint_path is None:
            ckpt_path = os.path.join(args.output_dir, args.name, "current_checkpoint")
        else:
            ckpt_path = args.checkpoint_path

        # check if exists current checkpoint path and it's not empty
        if not os.path.exists(ckpt_path) or len(os.listdir(ckpt_path)) == 0:
            console.log(
                f"[cyan]Checkpoint path {ckpt_path} does not exist or is empty. Cannot continue training.[/cyan]"
            )
            console.log(f"[cyan]Starting training from scratch.[/cyan]")
            return model, -1

        check_point = load_checkpoint(path=ckpt_path, rank=local_rank)
        state_dict = check_point["model_state_dict"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            console.log(
                f"[yellow]Missing keys in checkpoint: {len(missing_keys)} keys[/yellow]"
            )
        if unexpected_keys:
            console.log(
                f"[yellow]Unexpected keys in checkpoint (ignored): {len(unexpected_keys)} keys[/yellow]"
            )
            sample_unexpected = list(unexpected_keys)[:3]
            console.log(
                f"[yellow]Sample unexpected keys: {sample_unexpected}...[/yellow]"
            )
        start_step = check_point["global_step"]
    else:
        start_step = -1

    return model, start_step


def extract_metadata_from_graph(dataset: Data):
    # Get 1 graph from the dataset to make the GNN model become heterogeneous
    if "train" in dataset.processed_data:
        data_path = dataset.processed_data["train"][
            list(dataset.processed_data["train"].keys())[0]
        ]["path"]
    else:
        data_path = dataset.processed_data["test_module"][
            list(dataset.processed_data["test_module"].keys())[0]
        ]["path"]
    with open(data_path, "r") as f:
        sample = json.load(f)
    graph_path = sample["graph_path"]
    # Load graph with PyG classes allowlisted
    if graph_path is not None:
        with torch.serialization.safe_globals(
            [HeteroData, BaseStorage, GlobalStorage, NodeStorage, EdgeStorage]
        ):
            graph = torch.load(graph_path, weights_only=True)
    else:
        graph = None
    return graph.metadata()

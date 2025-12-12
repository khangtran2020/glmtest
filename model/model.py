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
        return get_model_train(
            args=args,
            console=console,
            tokenizer=tokenizer,
            rank=rank,
            metadata=metadata,
            use_zero3=use_zero3,
        )
    elif args.mode == "test":
        return get_model_test(
            args=args,
            console=console,
            tokenizer=tokenizer,
            rank=rank,
            metadata=metadata,
        )
    elif args.mode == "testgen":
        return get_model_testgen(
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

        if metadata is not None:
            glmf_model.gnn = to_hetero(glmf_model.gnn, metadata=metadata, aggr="sum")

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
        if args.model_weight_path is not None:

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

            model = GLMFModelForCausalLM(
                config=config,
                tokenizer=tokenizer,
                baseline_prompt=args.baseline_prompt,
                multi_gpu=True if args.num_gpu > 1 else False,
                debug=args.debug,
                rank=rank,
                is_training=False,
            )

            if metadata is not None:
                model.gnn = to_hetero(model.gnn, metadata=metadata, aggr="sum")

            for file in os.listdir(args.model_weight_path):
                if file.endswith(".pt"):
                    state_dict = torch.load(
                        os.path.join(args.model_weight_path, file),
                        map_location="cpu",
                        weights_only=True,
                    )
                    model.load_state_dict(state_dict)
                    console.log(
                        f"[red]Model weights loaded from {os.path.join(args.model_weight_path, file)}[/red]"
                    )

            model.init_for_train(tokenizer=tokenizer, rank=rank)
            # make the model to bf16/fp16
            for n, p in model.named_parameters():
                if args.dtype == "bf16":
                    if p.dtype != torch.bfloat16:
                        p.data = p.data.to(torch.bfloat16)
                elif args.dtype == "fp16":
                    if p.dtype != torch.float16:
                        p.data = p.data.to(torch.float16)

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
                    None
                    if use_zero3
                    else ("cuda" if torch.cuda.is_available() else "cpu")
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

            if metadata is not None:
                model.gnn = to_hetero(model.gnn, metadata=metadata, aggr="sum")

        model.llm_model.gradient_checkpointing_enable()
        model.config.graph_token_id = [
            tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
        ]

        console.log(
            f"Attention implementation of the model is: {model.llm_model.config._attn_implementation}"
        )
    return model


def get_model_test(
    args: Namespace,
    console: Console,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    metadata: tuple = None,
):
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
            tokenizer=tokenizer,
            baseline_prompt=args.baseline_prompt,
            debug=args.debug,
            rank=rank,
            multi_gpu=True if args.num_gpu > 1 else False,
            is_training=False,
        )

        if metadata is not None:
            glmf_model.gnn = to_hetero(glmf_model.gnn, metadata=metadata, aggr="sum")

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

        if metadata is not None:
            model.gnn = to_hetero(model.gnn, metadata=metadata, aggr="sum")

        model.config.graph_token_id = [
            tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
            tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
        ]

    # take .pt file from the model_weight_path
    console.log(f"[red]Finding weights from {args.model_weight_path}[/red]")
    for file in os.listdir(args.model_weight_path):
        if file.endswith(".pt"):
            state_dict = torch.load(
                os.path.join(args.model_weight_path, file),
                map_location=f"cuda:{rank}" if args.num_gpu > 1 else "cpu",
            )
            model.load_state_dict(state_dict)
            if use_lora:
                model.llm_model = model.llm_model.merge_and_unload()
            console.log(f"[red]Model weights loaded from {file}[/red]")

    # model = model.to(dtype=torch.float)
    # unifying dtype to avoid errors
    for n, p in model.named_parameters():
        if args.dtype == "bf16":
            if p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
        elif args.dtype == "fp16":
            if p.dtype != torch.float16:
                p.data = p.data.to(torch.float16)

    console.log(f"Model is loaded to device: {model.device} - with type {model.dtype}")
    for name, param in model.named_parameters():
        console.log(f"[yellow]Parameter {name}, dtype: {param.dtype}[/yellow]")
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

    if "current_checkpoint" in args.model_weight_path:
        use_lora = True
    else:
        use_lora = False

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

            if metadata is not None:
                glmf_model.gnn = to_hetero(
                    glmf_model.gnn, metadata=metadata, aggr="sum"
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

            if metadata is not None:
                model.gnn = to_hetero(model.gnn, metadata=metadata, aggr="sum")

            model.config.graph_token_id = [
                tokenizer.convert_tokens_to_ids(GRAPH_START_TOKEN),
                tokenizer.convert_tokens_to_ids(GRAPH_PAD_TOKEN),
                tokenizer.convert_tokens_to_ids(GRAPH_END_TOKEN),
            ]
        console.log(f"[red]Finding weights from {args.model_weight_path}[/red]")
        for file in os.listdir(args.model_weight_path):
            if file.endswith(".pt"):
                state_dict = torch.load(
                    os.path.join(args.model_weight_path, file),
                    map_location=f"cuda:{rank}" if args.num_gpu >= 1 else "cpu",
                )
                model.load_state_dict(state_dict)
                if use_lora:
                    model.llm_model = model.llm_model.merge_and_unload()
                console.log(f"[red]Model weights loaded from {file}[/red]")

        # model = model.to(dtype=torch.float)
        # unifying dtype to avoid errors
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
    else:
        model = None
    return model


def continue_training_from_checkpoint(
    args: Namespace,
    model: GLMFModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    local_rank: int,
):
    if args.continue_training:

        assert args.checkpoint_path is not None, "Checkpoint path must be specified."
        check_point = load_checkpoint(path=args.checkpoint_path, rank=local_rank)

        model.load_state_dict(check_point["model_state_dict"])
        optimizer.load_state_dict(check_point["optimizer_state_dict"])
        lr_scheduler.load_state_dict(check_point["scheduler_state_dict"])
        start_step = check_point["global_step"]
    else:
        start_step = -1

    return model, optimizer, lr_scheduler, start_step


def extract_metadata_from_graph(dataset: Data):
    # Get 1 graph from the dataset to make the GNN model become heterogeneous
    data_path = dataset["train"][list(dataset["train"].keys())[0]]["data_path"]
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

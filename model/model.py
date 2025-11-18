import os
import torch
from model.glmf import GLMFModelConfig, GLMFModelForCausalLM
from model.glmffuzz import GLMFModelFuzzing
from rich.console import Console
from argparse import Namespace
from transformers import PreTrainedTokenizer
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
):
    if args.mode == "train":
        return get_model_train(
            args=args, console=console, tokenizer=tokenizer, rank=rank
        )
    elif args.mode == "test":
        return get_model_test(
            args=args, console=console, tokenizer=tokenizer, rank=rank
        )
    elif args.mode == "testgen":
        return get_model_testgen(
            args=args,
            console=console,
            rank=rank,
            tokenizer=tokenizer,
            device=device,
        )
    else:
        console.log(f"Mode {args.mode} not using model.")
        return None


def get_model_train(
    args: Namespace, console: Console, tokenizer: PreTrainedTokenizer, rank: int
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
            gnn_type=args.gnn_type,
            multi_gpu=True if args.num_gpu > 1 else False,
            is_training=True,
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

        use_lora = False if args.only_gnn else args.use_lora

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
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

        model = GLMFModelForCausalLM(
            config=config,
            tokenizer=tokenizer,
            baseline_prompt=args.baseline_prompt,
            multi_gpu=True if args.num_gpu > 1 else False,
            debug=args.debug,
            rank=rank,
            gnn_type=args.gnn_type,
            is_training=True,
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

        if args.only_gnn:

            # freeze all params first
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze GNN params
            for name, param in model.named_parameters():
                if "gnn" in name:
                    param.requires_grad = True
                    console.log(f"Parameter {name} is set to be trainable.")

            # Count trainable parameters
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            console.log(
                f"[blue] Only GNN training mode activated. Total trainable parameters: {trainable_params} [/blue]"
            )

    if args.train_reasoning:
        console.log(f"[blue] Training with reasoning mode activated. [/blue]")
        if args.model_weight_path is not None:
            for file in os.listdir(args.model_weight_path):
                if file.endswith(".pt"):
                    state_dict = torch.load(
                        os.path.join(args.model_weight_path, file),
                        map_location=f"cuda:{rank}" if args.num_gpu > 1 else "cpu",
                    )
                    model.load_state_dict(state_dict, strict=False)
                    console.log(f"[red]Model weights loaded from {file}[/red]")

    return model


def get_model_test(
    args: Namespace,
    console: Console,
    tokenizer: PreTrainedTokenizer,
    rank: int,
):
    if rank == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{rank}")
    # load model
    assert (
        args.model_weight_path is not None
    ), "Model directory must be specified for testing."

    if (
        "current_checkpoint" in args.model_weight_path
        or "best_model" in args.model_weight_path
    ):
        use_lora = True
        console.log(f"[green]Using LoRA weights for testing.[/green]")
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
            gnn_type=args.gnn_type,
            multi_gpu=True if args.num_gpu > 1 else False,
            is_training=False,
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
            gnn_type=args.gnn_type,
            multi_gpu=True if args.num_gpu > 1 else False,
            is_training=False,
        )

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
            console.log(f"[green]Using LoRA weights for testing: {use_lora}.[/green]")
            # for key in list(state_dict.keys()):
            if "current_checkpoint" in args.model_weight_path:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict, strict=False)
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

    # console.log(f"Model is loaded to device: {model.device} - with type {model.dtype}")
    for name, param in model.named_parameters():
        if "gnn" in name:
            console.log(
                f"[yellow]Parameter {name}, dtype: {param.dtype}, devices: {param.device}[/yellow]"
            )
    return model


def get_model_testgen(
    args: Namespace,
    console: Console,
    rank: int,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
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
                gnn_type=args.gnn_type,
                multi_gpu=True if args.num_gpu > 1 else False,
                is_training=False,
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
                gnn_type=args.gnn_type,
                multi_gpu=True if args.num_gpu > 1 else False,
                is_training=False,
            )

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
                console.log(
                    f"[green]Using LoRA weights for testing: {use_lora}.[/green]"
                )
                # for key in list(state_dict.keys()):
                if "current_checkpoint" in args.model_weight_path:
                    state_dict = state_dict["model_state_dict"]
                model.load_state_dict(state_dict, strict=False)
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

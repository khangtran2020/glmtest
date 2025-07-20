import os
import gc
import torch
from torch import nn
from rich import print as pprint
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
import torch.distributed as dist
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.loss.loss_utils import fixed_cross_entropy

# from utils.prompter import Prompter
from model.gnn import MultiGAT
from model.utils.utils import create_causal_mask, create_sliding_window_causal_mask
from train.utils import extract_local
from ring_flash_attn import update_ring_flash_attn_params
from peft import get_peft_model, LoraConfig, TaskType
from peft.tuners.lora.model import LoraModel
from utils.constant import FUZZ_START_TOKEN, FUZZ_END_TOKEN

# VAE
from model.layer import GLMFFuzzingLayer

# typing
from accelerate import Accelerator
from typing import Callable, List, Optional, Tuple, Union, Dict, Any


class GLMFModelConfig(PretrainedConfig):

    model_type = "glmf"

    def __init__(
        self,
        llm_model: Optional[str] = None,
        mode: str = "branch",
        in_feats: int = 772,
        n_hidden: int = 512,
        n_layers: int = 4,
        num_head: int = 8,
        dropout: float = 0.2,
        dtype: str = "float32",
        device_map=None,
        # LoRA parameters
        use_lora: bool = False,
        lora_r: int = 4,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = None,
        debug: bool = False,
        **kwargs,
    ):
        # super().__init__(**kwargs)
        config = AutoConfig.from_pretrained(llm_model).to_dict()
        if llm_model is None:
            # print("1")
            raise ValueError("`llm_model` must be provided to GLMFModelConfig.")

        config = AutoConfig.from_pretrained(llm_model).to_dict()

        # if "_attn_implementation_autoset" in kwargs:
        #     config.pop("_attn_implementation_autoset", None)

        # for key in kwargs:
        #     if key in config.keys():
        #         config.pop(key, None)

        for key in list(kwargs):
            config.pop(key, None)

        super().__init__(**config, **kwargs)

        self.llm_model = llm_model
        self.model_name = (
            config["_name_or_path"]
            if "_name_or_path" in config.keys()
            else kwargs["_name_or_path"]
        )
        self.mode = mode
        self.hidden_size = (
            config["hidden_size"]
            if "hidden_size" in config.keys()
            else kwargs["hidden_size"]
        )
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.num_head = num_head
        self.dropout = dropout
        self.dtype = dtype
        self.device_map = device_map
        self.debug = debug

        if lora_target_modules is None:
            # This can be adjusted depending on your underlying model's architecture.
            lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
                "lm_head",
            ]
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules

        # self.dtype = dtype
        # self.graph_token_id = [92302, 92303, 92304]
        # super().__init__(**config, **kwargs)

    def to_diff_dict(self):
        # Instead of comparing with a default instance (which fails),
        # simply return the full dict.
        return self.to_dict()

    def _get_non_default_generation_parameters(self) -> Dict[str, Any]:
        """
        Gets the non-default generation parameters on the PretrainedConfig instance
        """
        non_default_generation_parameters = {}
        decoder_attribute_name = None

        # Composite models don't have a default config, use their decoder config as a fallback for default values
        # If no known pattern is matched, then `default_config = None` -> check against the global generation defaults
        try:
            # print("Using lora is set to", self.use_lora)
            default_config = self.__class__(
                llm_model=self.llm_model,
                mode=self.mode,
                in_feats=self.in_feats,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                num_head=self.num_head,
                dropout=self.dropout,
                dtype=self.dtype,
                device_map=self.device_map,
                use_lora=self.use_lora,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_target_modules=self.lora_target_modules,
                debug=self.debug,
            )
        except ValueError:
            decoder_config = self.get_text_config(decoder=True)
            if decoder_config is not self:
                default_config = decoder_config.__class__()
            else:
                default_config = None

        # If it is a composite model, we want to check the subconfig that will be used for generation
        self_decoder_config = (
            self
            if decoder_attribute_name is None
            else getattr(self, decoder_attribute_name)
        )

        for (
            parameter_name,
            default_global_value,
        ) in self._get_global_generation_defaults().items():
            if hasattr(self_decoder_config, parameter_name):
                is_default_in_config = is_default_generation_value = None
                parameter_value = getattr(self_decoder_config, parameter_name)
                # Three cases in which is okay for the model config to hold generation config parameters:
                # 1. The parameter is set to `None`, effectivelly delegating its value to the generation config
                if parameter_value is None:
                    continue
                # 2. If we have a default config, then the instance should hold the same generation defaults
                if default_config is not None:
                    is_default_in_config = parameter_value == getattr(
                        default_config, parameter_name
                    )
                # 3. if we don't have a default config, then the instance should hold the global generation defaults
                else:
                    is_default_generation_value = (
                        parameter_value == default_global_value
                    )

                is_non_default = (is_default_in_config is False) or (
                    is_default_in_config is None
                    and is_default_generation_value is False
                )
                if is_non_default:
                    non_default_generation_parameters[parameter_name] = getattr(
                        self_decoder_config, parameter_name
                    )

        return non_default_generation_parameters


class GLMFModel(PreTrainedModel):
    pass


class GLMFModelForCausalLM(GLMFModel, GenerationMixin):

    config_class = GLMFModelConfig

    def __init__(
        self,
        config: GLMFModelConfig,
        rank: int = 0,
        tokenizer: PreTrainedTokenizer = None,
        baseline_prompt: str = None,
        multi_gpu: bool = False,
        debug: bool = False,
        is_training: bool = False,
    ):

        super().__init__(config)

        self.baseline_prompt = baseline_prompt
        self.multi_gpu = multi_gpu
        self.debug = debug
        self.rank = rank
        self.is_training = is_training

        self.gnn = MultiGAT(
            config.mode,
            config.in_feats,
            config.n_hidden,
            config.hidden_size,
            config.n_layers,
            config.num_head,
            config.dropout,
        )
        if config.dtype == "fp16":
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16,
                device_map=f"cuda:{rank}",
                attn_implementation="flash_attention_2",
            )
        elif config.dtype == "bf16":
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                device_map=f"cuda:{rank}",
                attn_implementation="flash_attention_2",
            )
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                device_map=f"cuda:{rank}",
                attn_implementation="flash_attention_2",
            )

        if self.is_training:
            self.llm_model.resize_token_embeddings(len(tokenizer))
            self.config.vocab_size = len(tokenizer)
        else:
            self.llm_model.resize_token_embeddings(len(tokenizer))

        # LoRA init
        if config.use_lora:
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)

        gc.collect()
        torch.cuda.empty_cache()
        self.model_type = config.model_type

    def extract_embedding(
        self,
        input_ids: torch.Tensor,
        graph: Optional[dict],
        graph_mask: Optional[torch.Tensor],
        graph_token_index: Optional[torch.LongTensor],
        inputs_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)

        if (
            (graph is not None)
            and ("graph" in self.baseline_prompt)
            and (inputs_embeds.size(1) > 1)
        ):
            assert graph_mask is not None
            assert graph_token_index is not None

            graph_embeds = self.gnn(graph, graph_mask)
            graph_embeds = graph_embeds.to(inputs_embeds.device)
            assert (
                graph_embeds.shape
                == inputs_embeds[
                    0, graph_token_index[0] : (graph_token_index[-1] + 1), :
                ].shape
            ), f"Shape mismatch in assignment: graph embedding shape {graph_embeds.shape}, input embedding shape: {inputs_embeds.shape}, graph_token_index: {len(graph_token_index)}!"

            inputs_embeds[0, graph_token_index[0] : (graph_token_index[-1] + 1), :] = (
                graph_embeds
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        graph: Optional[dict] = None,
        graph_mask: Optional[torch.Tensor] = None,
        graph_token_index: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        step: int = 0,
        accelerator: Optional[Accelerator] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        inputs_embeds = self.extract_embedding(
            input_ids=input_ids,
            graph=graph,
            inputs_embeds=inputs_embeds,
            graph_mask=graph_mask,
            graph_token_index=graph_token_index,
        )

        if accelerator is not None:
            accelerator.wait_for_everyone()
        if self.multi_gpu:
            return self.forward_llm(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                labels=labels,
                step=step,
                accelerator=accelerator,
            )
        else:
            return self.llm_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=past_key_values,
                labels=labels,
            )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # return self.llm_model.prepare_inputs_for_generation(
        #     input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs
        # )
        inputs = self.llm_model.prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs
        )
        # carry your graph inputs through each generation step
        for key in ("graph", "graph_mask", "graph_token_index"):
            if key in kwargs:
                inputs[key] = kwargs[key]
        return inputs

    @staticmethod
    def _reorder_cache(self, past_key_values, beam_idx):
        # return self.llm_model._reorder_cache(past_key_values, beam_idx)
        if hasattr(past_key_values, "reorder_cache"):
            # For newer transformers versions with Cache object
            return past_key_values.reorder_cache(beam_idx)
        else:
            # For older transformers versions with list of tensors
            return tuple(
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                )
                for layer_past in past_key_values
            )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        state_dict = {}
        # Remove the middle model
        for key in self.state_dict().keys():
            # if not key.startswith("middle"):
            state_dict[key] = self.state_dict()[key]

        # Save the encoder-decoder model
        super().save_pretrained(
            save_directory,
            state_dict=state_dict,
            is_main_process=is_main_process,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs,
        )

    def forward_llm(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        step: int = 0,
        accelerator: Optional[Accelerator] = None,
        logits_to_keep: Union[int, slice] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        process_group = dist.group.WORLD
        ignore_index = -100

        if labels is not None:
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            labels = labels[..., 1:].contiguous()

        seq_len = inputs_embeds.shape[-2]
        rank = self.rank

        num_processes = dist.get_world_size()
        inputs_embeds, cu_seqlens_emb = extract_local(
            inputs_embeds, rank, num_processes, inputs_embeds.device
        )
        if labels is not None:
            labels, cu_seqlens_lab = extract_local(
                labels, rank, num_processes, labels.device
            )
            assert (
                cu_seqlens_emb - cu_seqlens_lab
            ).sum().item() == 0, (
                f"cu_seqlens_emb: {cu_seqlens_emb}, cu_seqlens_lab: {cu_seqlens_lab}"
            )

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=inputs_embeds.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(inputs_embeds.shape[0], -1)
            )

        position_ids, cu_seqlens_pos = extract_local(
            position_ids, rank, num_processes, inputs_embeds.device
        )
        assert (
            cu_seqlens_emb - cu_seqlens_pos
        ).sum().item() == 0, (
            f"cu_seqlens_emb: {cu_seqlens_emb}, cu_seqlens_pos: {cu_seqlens_pos}"
        )

        pprint(
            f"[yellow]Step {step} - rank {rank}[/yellow]: [cyan]cu_seqlens_emb: {cu_seqlens_emb} [/cyan]"
        )
        update_ring_flash_attn_params(
            cu_seqlens=cu_seqlens_emb, process_group=process_group
        )
        if accelerator is not None:
            accelerator.wait_for_everyone()

        if self.is_training:
            outputs = self.llm_model.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
        else:
            outputs = self.llm_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.llm_model.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            logits = logits.float()
            logits = logits.view(-1, self.config.vocab_size)
            labels = labels.view(-1)
            labels = labels.to(logits.device)
            loss = fixed_cross_entropy(
                logits,
                labels,
                num_items_in_batch=None,
                ignore_index=ignore_index,
                **kwargs,
            )
            # pprint(
            #     f"[yellow]Step {step} - rank {rank}[/yellow]: [cyan]loss: {loss}[/cyan], [green]logits shape: {logits}[/green], [blue]labels shape: {labels}[/blue]"
            # )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


CONFIG_MAPPING.register(key="glmf", value=GLMFModelConfig)
MODEL_FOR_CAUSAL_LM_MAPPING.register(key=GLMFModelConfig, value=GLMFModelForCausalLM)


class GLMFModelFuzzing(GLMFModel, GenerationMixin):
    """
    A model class for fuzzing the GLMFModelForCausalLM.
    This class is used to test the model's behavior with different inputs.
    """

    config_class = GLMFModelConfig

    def __init__(
        self,
        config: GLMFModelConfig,
        rank: int = 0,
        tokenizer: PreTrainedTokenizer = None,
        baseline_prompt: str = None,
        multi_gpu: bool = False,
        debug: bool = False,
        is_training: bool = False,
        layer_indices: List[int] = None,
        glmf_model: Optional[GLMFModelForCausalLM] = None,
        kl_g_reg: float = 0.0,
        kl_d_reg: float = 0.0,
        **kwargs,
    ):

        super().__init__(config)

        self.baseline_prompt = baseline_prompt
        self.multi_gpu = multi_gpu
        self.debug = debug
        self.rank = rank
        self.is_training = is_training
        self.kl_g_reg = kl_g_reg
        self.kl_d_reg = kl_d_reg
        self.tokenizer = tokenizer

        if glmf_model is not None:
            # If a GLMFModelForCausalLM is provided, use its configuration
            # config = glmf_model.config
            self.glmf_model = glmf_model
            self.gnn = glmf_model.gnn
            self.llm_model = glmf_model.llm_model
            self.rotary_emb = self.llm_model.model.rotary_emb
        else:
            raise ValueError(
                "A GLMFModelForCausalLM instance must be provided to GLMFModelFuzzing."
            )

        # LoRA init
        # print("Config use_lora:", config.use_lora)
        if config.use_lora:
            # print("Using LoRA for the model.")
            # If LoRA is enabled, we need to apply
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)

        print(f"Struture of the model: {self.llm_model}")

        if layer_indices is not None:
            # Patch the model with GLMFFuzzingLayer at the specified layer indices
            self.patch_model_with_fuzz_layer(layer_indices=layer_indices)

        if hasattr(self.llm_model, "base_model"):
            self.layers = self.llm_model.base_model.model.model.layers
        else:
            self.layers = self.llm_model.model.layers

        gc.collect()
        torch.cuda.empty_cache()
        self.model_type = config.model_type

    def patch_model_with_fuzz_layer(self, layer_indices: List[int]) -> None:
        """
        Patch the model with a GLMFFuzzingLayer at the specified layer index.
        This is used to test the model's behavior with fuzzing inputs.
        """
        if not hasattr(self.llm_model, "model") or (
            not hasattr(self.llm_model, "base_model")
        ):
            raise ValueError("The model does not have a valid structure for patching.")

        if hasattr(self.llm_model, "base_model"):
            # If the model has a base_model attribute, it is likely a LoRA model
            self._patch_peft_model(layer_indices)
        else:
            # If the model does not have a base_model attribute, it is likely a casual LM model
            self._patch_casual_lm_model(layer_indices)

    def _patch_peft_model(self, layer_indices: List[int]) -> None:

        for layer_index in layer_indices:
            if layer_index < 0 or layer_index >= len(
                self.llm_model.base_model.model.model.layers
            ):
                raise IndexError(
                    f"Layer index {layer_index} is out of bounds for the LoRA model's layers."
                )
            # Patch the layer
            self.llm_model.base_model.model.model.layers[layer_index] = (
                GLMFFuzzingLayer(
                    d_model=self.config.hidden_size,
                    nhead=self.config.num_head,
                    llm_layer=self.llm_model.base_model.model.model.layers[layer_index],
                    dim_feedforward=self.config.n_hidden,
                    dropout=self.config.dropout,
                    is_fuzz=True,
                )
            )

    def _patch_casual_lm_model(self, layer_indices: List[int]) -> None:
        """
        Patch the model with a GLMFFuzzingLayer at the specified layer index.
        This is used to test the model's behavior with fuzzing inputs.
        """

        for layer_index in layer_indices:
            if layer_index < 0 or layer_index >= len(self.llm_model.model.layers):
                raise IndexError(
                    f"Layer index {layer_index} is out of bounds for the model's layers."
                )
            # Patch the layer
            self.llm_model.model.layers[layer_index] = GLMFFuzzingLayer(
                d_model=self.config.hidden_size,
                nhead=self.config.num_head,
                llm_layer=self.llm_model.model.layers[layer_index],
                dim_feedforward=self.config.n_hidden,
                dropout=self.config.dropout,
                is_fuzz=True,
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        graph: Optional[dict] = None,
        graph_mask: Optional[torch.Tensor] = None,
        # fuzzing_mask: Optional[torch.Tensor] = None,
        graph_token_index: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        step: int = 0,
        accelerator: Optional[Accelerator] = None,
        **flash_attn_kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # extract the fuzzing mask from labels.
        if labels is not None:
            # get token id from tokenizer
            fuzz_start_id = self.tokenizer.convert_tokens_to_ids(FUZZ_START_TOKEN)
            fuzz_end_id = self.tokenizer.convert_tokens_to_ids(FUZZ_END_TOKEN)
            fuzzing_mask = torch.zeros(labels.shape, device=labels.device)
            for i in range(labels.shape[0]):
                saw_start = False
                for j in range(labels.shape[1]):
                    if saw_start:
                        fuzzing_mask[i, j] = 1

                    if labels[i, j] == fuzz_start_id:
                        saw_start = True
                    elif labels[i, j] == fuzz_end_id:
                        saw_start = False
            fuzzing_mask = fuzzing_mask.unsqueeze(-1)
        else:
            fuzzing_mask = None

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        inputs_embeds = self.extract_embedding(
            input_ids=input_ids,
            graph=graph,
            inputs_embeds=inputs_embeds,
            graph_mask=graph_mask,
            graph_token_index=graph_token_index,
        )

        if position_ids is None:
            position_ids = (
                torch.arange(
                    inputs_embeds.size(1), device=inputs_embeds.device, dtype=torch.long
                )
                .unsqueeze(0)
                .expand(inputs_embeds.shape[0], -1)
            )

        if self.config.model_type == "qwen2":
            if not isinstance(causal_mask_mapping := attention_mask, dict):
                pprint(
                    "[green]Using Qwen2 model type for forward pass. Creating causal_mask[/green]"
                )
                # Prepare mask arguments
                mask_kwargs = {
                    "config": self.llm_model.config,
                    "input_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                }
                # Create the masks
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                # The sliding window alternating layers are not always activated depending on the config
                if hasattr(self.llm_model, "base_model"):
                    if self.llm_model.base_model.model.model.has_sliding_layers:
                        causal_mask_mapping["sliding_attention"] = (
                            create_sliding_window_causal_mask(**mask_kwargs)
                        )
                else:
                    if self.llm_model.model.has_sliding_layers:
                        causal_mask_mapping["sliding_attention"] = (
                            create_sliding_window_causal_mask(**mask_kwargs)
                        )

        elif self.config.model_type == "llama":
            causal_mask = create_causal_mask(
                config=self.llm_model.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
            )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        latent_dict = None
        kl_d_total = 0.0
        kl_g_total = 0.0

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.config.model_type == "qwen2":
                pprint("[green]Casual mask mapping:[/green]", causal_mask_mapping)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[
                        (
                            decoder_layer.llm_layer.attention_type
                            if isinstance(decoder_layer, GLMFFuzzingLayer)
                            else decoder_layer.attention_type
                        )
                    ],
                    fuzzing_mask=fuzzing_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    latent_dict=latent_dict,
                    **flash_attn_kwargs,
                )
            elif self.config.model_type == "llama":
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            if isinstance(decoder_layer, GLMFFuzzingLayer):
                hidden_states, attention_out, kl_g, kl_d, latent_dict = layer_outputs
                if kl_g is not None:
                    kl_g_total += kl_g
                if kl_d is not None:
                    kl_d_total += kl_d
                if output_attentions:
                    all_self_attns += (attention_out,)
            else:
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        if hasattr(self.llm_model, "base_model"):
            hidden_states = self.llm_model.base_model.model.model.norm(hidden_states)
        else:
            hidden_states = self.llm_model.model.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # logits = self.lm_head(hidden_states)
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        if hasattr(self.llm_model, "base_model"):
            logits = self.llm_model.base_model.lm_head(
                hidden_states[:, slice_indices, :]
            )
        else:
            logits = self.llm_model.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:

            logits = logits.float()
            logits = logits.view(-1, self.config.vocab_size)
            labels = labels.view(-1)
            labels = labels.to(logits.device)
            loss = fixed_cross_entropy(
                logits,
                labels,
                num_items_in_batch=None,
                ignore_index=-100,
            )
            # if hasattr(self.llm_model, "base_model"):
            #     self.llm_model.base_model.loss_function(
            #         logits=logits,
            #         labels=labels,
            #         vocab_size=self.config.vocab_size,
            #     )
            # else:
            #     self.llm_model.loss_function(
            #         logits=logits,
            #         labels=labels,
            #         vocab_size=self.config.vocab_size,
            #     )
            loss = loss + self.kl_g_reg * kl_g_total + self.kl_d_reg * kl_d_total

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_self_attns if output_attentions else None,
        )

    def extract_embedding(
        self,
        input_ids: torch.Tensor,
        graph: Optional[dict],
        graph_mask: Optional[torch.Tensor],
        graph_token_index: Optional[torch.LongTensor],
        inputs_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)

        if (
            (graph is not None)
            and ("graph" in self.baseline_prompt)
            and (inputs_embeds.size(1) > 1)
        ):
            assert graph_mask is not None
            assert graph_token_index is not None

            graph_embeds = self.gnn(graph, graph_mask)
            graph_embeds = graph_embeds.to(inputs_embeds.device)
            assert (
                graph_embeds.shape
                == inputs_embeds[
                    0, graph_token_index[0] : (graph_token_index[-1] + 1), :
                ].shape
            ), f"Shape mismatch in assignment: graph embedding shape {graph_embeds.shape}, input embedding shape: {inputs_embeds.shape}, graph_token_index: {len(graph_token_index)}!"

            inputs_embeds[0, graph_token_index[0] : (graph_token_index[-1] + 1), :] = (
                graph_embeds
            )

        return inputs_embeds

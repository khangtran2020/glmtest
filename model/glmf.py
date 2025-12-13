import os
import time
import torch
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
from model.gnn import GAT, SAGE
from train.utils import extract_local
from peft import get_peft_model, LoraConfig, TaskType
from utils.utils import get_index_by_value


# typing
from accelerate import Accelerator
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
from torch_geometric.data import HeteroData

# from ring_flash_attn import update_ring_flash_attn_params


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
        use_lora: bool = False,
        lora_r: int = 4,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = None,
        debug: bool = False,
        use_flash_attn: bool = False,
        **kwargs,
    ):
        # super().__init__(**kwargs)
        config = AutoConfig.from_pretrained(llm_model).to_dict()
        if llm_model is None:
            # print("1")
            raise ValueError("`llm_model` must be provided to GLMFModelConfig.")

        config = AutoConfig.from_pretrained(llm_model).to_dict()

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
        self.use_flash_attn = use_flash_attn

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

    def to_diff_dict(self):
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
        gnn_type: str = "gat",
        tokenizer: PreTrainedTokenizer = None,
        baseline_prompt: str = None,
        multi_gpu: bool = False,
        debug: bool = False,
        is_training: bool = False,
        use_zero3: bool = False,
    ):

        super().__init__(config)

        self.baseline_prompt = baseline_prompt
        self.multi_gpu = multi_gpu
        self.debug = debug
        self.rank = rank
        self.is_training = is_training
        self.gnn_mode = config.mode
        self.use_lora = config.use_lora

        if "graph" in self.baseline_prompt:
            if gnn_type == "gat":
                self.gnn: GAT = GAT(
                    in_feats=config.in_feats,
                    n_hidden=config.n_hidden,
                    hidden_size=config.hidden_size,
                    n_layers=config.n_layers,
                    num_head=config.num_head,
                    dropout=config.dropout,
                )
            else:  # graphsage
                self.gnn: SAGE = SAGE(
                    in_feats=config.in_feats,
                    n_hidden=config.n_hidden,
                    hidden_size=config.hidden_size,
                    n_layers=config.n_layers,
                    dropout=config.dropout,
                )

        if config.dtype == "fp16":
            torch_dtype = torch.float16
        elif config.dtype == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = None

        # For ZeRO-3, device_map must be None as DeepSpeed handles device placement
        model_kwargs = {
            "dtype": torch_dtype,
            "attn_implementation": (
                "flash_attention_2" if config.use_flash_attn else "sdpa"
            ),
        }
        if not use_zero3:
            model_kwargs["device_map"] = f"cuda:{rank}"

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs,
        )

        if self.is_training:
            self.llm_model.resize_token_embeddings(len(tokenizer))
            self.config.vocab_size = len(tokenizer)
        else:
            self.llm_model.resize_token_embeddings(len(tokenizer))

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

        self.model_type = config.model_type

    def extract_embedding(
        self,
        input_ids: torch.Tensor,
        graphs: Optional[List[HeteroData]],
        graph_masks: Optional[List[torch.Tensor]],
        graph_token_indices: Optional[List[torch.LongTensor]],
        inputs_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        embedding_start = time.time()

        # embeds = []
        if inputs_embeds is None:
            # pprint(f"[blue]Input ids shape: {input_ids.shape}[/blue]")
            input_ids = input_ids.to(self.llm_model.device)
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
            input_ids = input_ids.to("cpu")

        if (
            (graphs is not None)
            and ("graph" in self.baseline_prompt)
            and (inputs_embeds.size(1) > 1)
        ):
            assert graph_masks is not None
            assert graph_token_indices is not None

            batch_size = inputs_embeds.size(0)
            for i in range(batch_size):

                graph_token_index = graph_token_indices[i]
                graph_mask = graph_masks[i]
                graph_mask = [mask for mask in graph_mask if mask.sum() > 0]

                if self.gnn_mode == "node":
                    ranges = []
                    start = graph_token_index[0]
                    prev = graph_token_index[0]
                    for j in graph_token_index[1:]:
                        if j == prev + 1:
                            prev = j
                        else:
                            ranges.append((start, prev))
                            start = j
                            prev = j
                    ranges.append((start, prev))

                    assert len(ranges) == len(
                        graph_mask
                    ), f"Mismatch between graph masks {len(graph_mask)} and token index ranges {len(ranges)}. Graph token index are ranges: {ranges}."

                graph = graphs[i]
                overall_mask = None  # merge graph_mask
                for j, mask in enumerate(graph_mask):
                    if j == 0:
                        overall_mask = mask.to(torch.bool)
                    else:
                        overall_mask = overall_mask | mask.to(torch.bool)
                overall_indices = (overall_mask == 1).nonzero(as_tuple=True)[0]
                mask_idx = []
                for j, mask in enumerate(graph_mask):
                    mask_indices = (mask == 1).nonzero(as_tuple=True)[0]
                    idx_in_overall = []
                    for k, idx in enumerate(mask_indices):
                        if idx in overall_indices:
                            idx_in_overall.append(k)
                    mask_idx.append(idx_in_overall)

                overall_mask = overall_mask.long().to(self.llm_model.device)
                graph = graph.to(self.llm_model.device)

                for node_type in graph.node_types:
                    if "x" in graph[node_type]:
                        graph[node_type].x = graph[node_type].x.half()

                graph_embeds = self.gnn(graph.x_dict, graph.edge_index_dict)
                graph_embeds = graph_embeds["node"][overall_indices, :]

                if self.gnn_mode == "node":
                    for j, mask in enumerate(graph_mask):
                        embeds = graph_embeds[mask_idx[j], :]
                        assert embeds.size(0) == len(mask_idx[j])
                        embeds = embeds.to(inputs_embeds.device)
                        inputs_embeds[i, ranges[j][0] : ranges[j][1] + 1, :] = (
                            embeds.to(inputs_embeds.dtype)
                        )
                else:  # branch mode
                    for j, mask in enumerate(graph_mask):
                        embeds = graph_embeds[mask_idx[j], :]
                        assert embeds.size(0) == len(mask_idx[j])
                        inputs_embeds[
                            i, graph_token_index[j] : graph_token_index[j] + 1, :
                        ] = embeds.to(inputs_embeds.dtype).mean(dim=0, keepdim=True)
        else:
            if self.is_training:
                inputs_embeds = inputs_embeds.requires_grad_(True)
        self._last_embedding_time = time.time() - embedding_start
        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        graphs: Optional[List[dict]] = None,
        graph_masks: Optional[List[torch.Tensor]] = None,
        graph_token_indices: Optional[List[torch.LongTensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # step: int = 0,
        # ring_attn: bool = False,
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
            graphs=graphs,
            inputs_embeds=inputs_embeds,
            graph_masks=graph_masks,
            graph_token_indices=graph_token_indices,
        )

        # pprint(f"[green]Input embeds shape: {inputs_embeds.shape}[/green]")

        if accelerator is not None:
            accelerator.wait_for_everyone()
        if self.multi_gpu:
            # if not ring_attn:
            return self.llm_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=past_key_values,
                labels=labels,
            )
            # else:
            #     return self.forward_llm(
            #         input_ids=None,
            #         inputs_embeds=inputs_embeds,
            #         position_ids=position_ids,
            #         attention_mask=attention_mask,
            #         use_cache=use_cache,
            #         labels=labels,
            #         step=step,
            #         accelerator=accelerator,
            #     )
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

    """def forward_llm(
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

        # print("Input embeds requires_grad:", inputs_embeds.requires_grad)

        if labels is not None:
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            labels = labels[..., 1:].contiguous()

        seq_len = inputs_embeds.shape[-2]
        rank = self.rank

        num_processes = dist.get_world_size()
        # pprint(f"[blue]Number of processes: {num_processes}[/blue]")
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
        update_ring_flash_attn_params(
            cu_seqlens=cu_seqlens_emb, process_group=process_group
        )
        if accelerator is not None:
            accelerator.wait_for_everyone()

        if self.is_training:
            if self.use_lora:
                outputs = self.llm_model.base_model.model.model(
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
        if self.is_training and self.use_lora:
            logits = self.llm_model.base_model.model.lm_head(
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
                ignore_index=ignore_index,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )"""

    def init_for_train(self, tokenizer: PreTrainedTokenizer):

        # if self.is_training:
        self.is_training = True
        self.llm_model.resize_token_embeddings(len(tokenizer))
        self.config.vocab_size = len(tokenizer)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm_model = get_peft_model(self.llm_model, lora_config)


CONFIG_MAPPING.register(key="glmf", value=GLMFModelConfig)
MODEL_FOR_CAUSAL_LM_MAPPING.register(key=GLMFModelConfig, value=GLMFModelForCausalLM)

import os
import gc
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

# from utils.prompter import Prompter
from train.utils import run_nvidia_smi
from model.gnn import MultiGAT
from train.utils import extract_local
from peft import get_peft_model, LoraConfig, TaskType

# typing
from typing import Callable, List, Optional, Tuple, Union


class GLMFModelConfig(PretrainedConfig):

    def __init__(
        self,
        llm_model: str,
        mode: str = "node",
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
        self.model_name = config["_name_or_path"]
        self.mode = mode
        self.hidden_size = config["hidden_size"]
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
        super().__init__(**config, **kwargs)

    def to_diff_dict(self):
        # Instead of comparing with a default instance (which fails),
        # simply return the full dict.
        return self.to_dict()


class GLMFModel(PreTrainedModel):
    pass


class GLMFModelForCausalLM(GLMFModel, GenerationMixin):

    config_class = GLMFModelConfig

    def __init__(
        self,
        config: GLMFModelConfig,
        rank: int,
        tokenizer: PreTrainedTokenizer = None,
        baseline_prompt: str = None,
        multi_gpu: bool = False,
        debug: bool = False,
    ):

        super().__init__(config)

        self.baseline_prompt = baseline_prompt
        self.multi_gpu = multi_gpu
        self.debug = debug
        self.rank = rank

        self.gnn = MultiGAT(
            config.mode,
            config.in_feats,
            config.n_hidden,
            config.hidden_size,
            config.n_layers,
            config.num_head,
            config.dropout,
        )
        if config.dtype == "float16":
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16,
                device_map=config.device_map,
            )
        elif config.dtype == "bfloat16":
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                device_map=config.device_map,
            )
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                device_map=config.device_map,
            )

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
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        step: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if self.debug:
            print("=" * 100 + f"Rank {self.rank} - Step {step}" + "\n\n")
            print(
                f"Before take graph embedding, size of inputs_embeds: {input_ids.size()}"
            )
            print("\n\n" + "=" * 100)
            # run_nvidia_smi(console=None)

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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)

        if (graph is not None) and ("graph" in self.baseline_prompt):
            assert graph_mask is not None
            assert graph_token_index is not None

            graph_embeds = self.gnn(graph, graph_mask)
            graph_embeds = graph_embeds.to(inputs_embeds.device)
            assert (
                graph_embeds.shape
                == inputs_embeds[
                    0, graph_token_index[0] : (graph_token_index[-1] + 1), :
                ].shape
            ), "Shape mismatch in assignment!"

            if self.debug:
                print(
                    "Printing the size of inputs_embeds, and graph_embeds",
                    inputs_embeds.size(),
                    graph_embeds.size(),
                )

            inputs_embeds[0, graph_token_index[0] : (graph_token_index[-1] + 1), :] = (
                graph_embeds
            )

        if self.debug:
            print("=" * 100 + f"Rank {self.rank} - Step {step}" + "\n\n")
            print(
                f"After take graph embedding, size of inputs_embeds: {inputs_embeds.size()}"
            )
            print("\n\n" + "=" * 100)

        if self.debug:
            print("After take graph embedding")
            run_nvidia_smi(console=None)
        # print(inputs_embeds.size())
        if self.multi_gpu:
            return self.forward_llm(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=False,
                labels=labels,
                step=step,
            )
        else:
            return self.llm_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=False,
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
        return self.llm_model.prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs
        )

    @staticmethod
    def _reorder_cache(self, past_key_values, beam_idx):
        return self.llm_model._reorder_cache(past_key_values, beam_idx)

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        seq_len = inputs_embeds.shape[-2]
        rank = self.rank
        num_processes = dist.get_world_size()
        inputs_embeds = extract_local(
            inputs_embeds, rank, num_processes, inputs_embeds.device
        )
        if self.debug:
            print(
                f"Rank {rank} inputs_embeds at step {step}: {inputs_embeds.size()}, device: {inputs_embeds.device}"
            )
        if labels is not None:
            labels = extract_local(labels, rank, num_processes, labels.device)
        position_ids = (
            torch.arange(seq_len, device=inputs_embeds.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(inputs_embeds.shape[0], -1)
        )
        position_ids = extract_local(
            position_ids, rank, num_processes, position_ids.device
        )

        return self.llm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

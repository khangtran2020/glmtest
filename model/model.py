import os
import torch

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

# from utils.prompter import Prompter
from model.gnn import MultiGAT
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
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self.graph_token_id = [92302, 92303, 92304]
        super().__init__(**config, **kwargs)


class GLMFModel(PreTrainedModel):
    pass


class GLMFModelForCausalLM(GLMFModel, GenerationMixin):

    config_class = GLMFModelConfig

    def __init__(self, config: GLMFModelConfig):

        super().__init__(config)

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
                # load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map=config.device_map,
            )
        elif config.dtype == "bfloat16":
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                # load_in_8bit=True,
                torch_dtype=torch.bfloat16,
                device_map=config.device_map,
            )
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                # load_in_8bit=True,
                device_map=config.device_map,
            )

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

        # del self.model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        self.model_type = config.model_type

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        graph: Optional[dict] = None,
        graph_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print("Start")
        # print(inputs_embeds)

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
            # inputs_embeds = self.model.model.embed_tokens(input_ids)
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
            print(inputs_embeds.device)
            # print("Test Inputs Before")
            print(inputs_embeds.size())

        if graph is not None:
            assert graph_mask is not None
            # print("Test Graph")
            index = torch.where(input_ids == self.config.graph_token_id[1])[1]
            graph_embeds = self.gnn(graph, graph_mask).to(inputs_embeds.device)
            # graph_embeds = self.gnn(graph, graph_mask)
            # print(graph_embeds)
            print("Graph_embeds: ", graph_embeds.size())
            # assert graph_embeds.size(2) == inputs_embeds.size(2)

            inputs_embeds[0, index[0] : (index[-1] + 1), :] = graph_embeds
            del graph_embeds

        # print(inputs_embeds.size())

        output = self.llm_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
        )
        return output

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

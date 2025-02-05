from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig
import os
from typing import Callable, List, Optional, Tuple, Union
from transformers import AutoModel, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
# from .config import XCodeConfig
import torch.nn as nn
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers import set_seed
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
# from utils.prompter import Prompter
import transformers
from gnn import MultiGAT


class XCodeConfig(PretrainedConfig):
    model_type = "xcode"

    def __init__(
        self,
        vlmodel = f"../Qwen2-VL-7B-Testcase",
        mode = "node",
        in_feats=772,
        n_hidden=512,
        n_layers=4,
        num_head=8,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        vlconfig = AutoConfig.from_pretrained(vlmodel).to_dict()
        self.model_name = vlconfig["_name_or_path"]
        self.mode = mode
        self.hidden_size = vlconfig["hidden_size"]
        self.in_feats = in_feats
        self.n_hidden= n_hidden        
        self.n_layers = n_layers
        self.num_head = num_head
        self.dropout = dropout
        self.graph_token_id = [92302, 92303,92304]
        super().__init__(**vlconfig ,**kwargs)


class XCodeModel(PreTrainedModel):
    pass

class XCodeModelForCausalLM(XCodeModel, GenerationMixin):
    config_class = XCodeConfig

    def __init__(self, config: XCodeConfig):
        super().__init__(config)
        
        self.gnn = MultiGAT(config.mode, config.in_feats, config.n_hidden, config.hidden_size, config.n_layers, config.num_head, config.dropout)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name, device_map="auto"
        )

        import gc  # garbage collect library

        gc.collect()
        torch.cuda.empty_cache()
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
        print("Start")
        print(inputs_embeds)

    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.model.model.embed_tokens(input_ids)
            print("Test Inputs Before")
            print(inputs_embeds.size())
            print(inputs_embeds)

        if graph is not None:
            assert graph_mask is not None
            print("Test Graph")
            index = torch.where(input_ids == self.config.graph_token_id[1])[1]
            graph_embeds = self.gnn(graph, graph_mask).to(inputs_embeds.device)
            print(graph_embeds)
            print(graph_embeds.size())
            # assert graph_embeds.size(2) == inputs_embeds.size(2)
            print(index)

        print(inputs_embeds)
        left = inputs_embeds[:, :index, :]   # Shape: (batch_size, index, embedding_dim)
        print(left.size())
        right = inputs_embeds[:, index:, :]   # Shape: (batch_size, seq_length-index, embedding_dim)
        print(right.size())
        
        graph_embeds = graph_embeds.unsqueeze(0)
        print(graph_embeds.size())
        
        inputs_embeds = torch.cat((left, graph_embeds, right), dim=1)
        # inputs_embeds[0, index, :] = graph_embeds
        print("Test Inputs After")
        print(inputs_embeds.size())
        print(inputs_embeds)
        
        attention_mask = torch.ones(inputs_embeds.size()).to(self.model.device)
        hidden_states = inputs_embeds
        print("Position Id")
        print(position_ids)
        print(position_ids.size())

        position_ids = torch.arange(inputs_embeds.size(1)).unsqueeze(0).to(self.model.device)
        print(position_ids)
        print(position_ids.size())
        
        output = self.model(
            inputs_embeds=inputs_embeds.to(self.model.device),
            # inputs_embeds = hidden_states.to(self.model.device),
            hidden_states = hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
            graph=None,
        )
        print("Output")
        print(output)
        # hidden_states = output[0]

        return output

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        return self.model.prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs)


    @staticmethod
    def _reorder_cache(self,past_key_values, beam_idx):
      return self.model._reorder_cache(past_key_values, beam_idx)

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
            #if not key.startswith("middle"):
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
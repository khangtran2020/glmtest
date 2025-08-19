"""
Inherited from NVIB Transformer Encoder Layer: https://github.com/idiap/nvib_selfattention
"""

import torch
import torch.nn.functional as F
from model.nvib.nvib.denoising_attention import DenoisingMultiheadAttention
from model.nvib.nvib.nvib_layer import Nvib
from torch import Tensor
from torch.nn.modules import Dropout, LayerNorm, Linear, Module
from torch.nn.modules.transformer import _get_activation_fn, _get_clones
from torch.nn import Module, Linear, Dropout, LayerNorm
from transformers import Cache
from rich import print as pprint

# typings
from typing import Callable, Optional, Union, Tuple


class NVIBTransformerLayer(Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        device=None,
        dtype=None,
        kappa=1.0,
        delta=1.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NVIBTransformerLayer, self).__init__()
        self.nvib_layer = Nvib(
            size_in=d_model,
            size_out=d_model,
            delta=delta,
            kappa=kappa,
            nheads=nhead,
        )
        self.self_attn = DenoisingMultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, **factory_kwargs
        )

        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(NVIBTransformerLayer, self).__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        latent_dict=None,
        fuzzing_mask: Optional[Tensor] = None,
    ) -> Tensor:

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ""
        if not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src_mask is not None:
            why_not_sparsity_fast_path = "src_mask is not supported for fastpath"
        elif src.is_nested and src_key_padding_mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask is not supported with NestedTensor input for fastpath"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or "cpu" in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "some Tensor argument is neither CUDA nor CPU"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if not why_not_sparsity_fast_path:
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    # TODO: if src_mask and src_key_padding_mask merge to single 4-dim mask
                    src_mask if src_mask is not None else src_key_padding_mask,
                    (
                        1
                        if src_key_padding_mask is not None
                        else 0 if src_mask is not None else None
                    ),
                )

        x = src
        # Check nan in the input
        if torch.isnan(x).any():
            print("NaN detected in input x")
            pprint(x)
        # Alpha skip
        if latent_dict is not None:
            alpha_skip = latent_dict["alpha"]
        else:
            alpha_skip = None

        # Nvib latent dictionary
        out, attention, latent_dict = self._sa_block(
            x, src_mask, src_key_padding_mask, alpha_skip
        )
        x = x + out
        x = x + self._ff_block(self.norm2(x))

        # Calculate KL divergence
        for key in latent_dict.keys():
            if isinstance(latent_dict[key], Tensor):
                print(f"{key} shape:", latent_dict[key].shape)
            elif isinstance(latent_dict[key], tuple):
                print(f"{key} shape:", tuple(t.shape for t in latent_dict[key]))

        latent_dict["memory_key_padding_mask"] = latent_dict[
            "memory_key_padding_mask"
        ].transpose(1, 0)
        latent_dict["fuzzing_mask"] = (
            fuzzing_mask.transpose(1, 0) if fuzzing_mask is not None else None
        )
        kl_g = self.nvib_layer.kl_gaussian(
            mu=latent_dict["mu"],
            logvar=latent_dict["logvar"],
            alpha=latent_dict["alpha"],
            memory_key_padding_mask=latent_dict["memory_key_padding_mask"],
            fuzzing_mask=fuzzing_mask,
        )
        kl_d = self.nvib_layer.kl_dirichlet(
            alpha=latent_dict["alpha"],
            memory_key_padding_mask=latent_dict["memory_key_padding_mask"],
            fuzzing_mask=fuzzing_mask,
        )
        return x, attention, kl_g, kl_d, latent_dict

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        alpha_skip=None,
    ) -> Tensor:
        # Note query does not include the prior
        latent_dict = self.nvib_layer(x, key_padding_mask, alpha_skip)
        query = x
        key = latent_dict["z"]
        value = latent_dict["z"]

        pprint("Latent dictionary:", latent_dict)
        # check nan in the query, key, value
        # if torch.isnan(query).any():
        #     print("NaN detected in query inside _sa_block")
        #     pprint(query)

        # if torch.isnan(key).any():
        #     print("NaN detected in key inside _sa_block")
        #     pprint(key)

        # if torch.isnan(value).any():
        #     print("NaN detected in value inside _sa_block")
        #     pprint(value)

        x, attention = self.self_attn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=latent_dict["memory_key_padding_mask"],
            need_weights=True,
        )
        return self.dropout2(x), attention, latent_dict

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class GLMFFuzzingLayer(Module):
    r"""
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = GLMFFuzzingLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = GLMFFuzzingLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """

    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        llm_layer: Module,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        device=None,
        dtype=None,
        kappa=1.0,
        delta=1.0,
        is_fuzz: bool = False,
    ) -> None:
        super(GLMFFuzzingLayer, self).__init__()
        self.llm_layer = llm_layer
        self.is_fuzz = is_fuzz
        if self.is_fuzz:
            self.nvib_layer = NVIBTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                device=device,
                dtype=dtype,
                kappa=kappa,
                delta=delta,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fuzzing_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        latent_dict: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        llm_hidden_state = self.llm_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if not self.is_fuzz:
            return llm_hidden_state

        # print(f"Fuzzing mask: {fuzzing_mask}")

        src_key_padding_mask = (
            ~(attention_mask.bool())
            if attention_mask is not None
            else torch.zeros(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
        )
        hidden_states = self.llm_layer.input_layernorm(hidden_states)
        nvib_hidden_states, attention_out, kl_g, kl_d, latent_dict_out = (
            self.nvib_layer(
                hidden_states,
                src_key_padding_mask=src_key_padding_mask,
                latent_dict=latent_dict,
            )
        )

        # Check nan in the outcome
        if torch.isnan(nvib_hidden_states).any():
            print("NaN detected in nvib_hidden_states")
            pprint(nvib_hidden_states)
        if torch.isnan(attention_out).any():
            print("NaN detected in attention_out")
            pprint(attention_out)
        if torch.isnan(kl_g).any():
            print("NaN detected in kl_g")
            pprint(kl_g)
        if torch.isnan(kl_d).any():
            print("NaN detected in kl_d")
            pprint(kl_d)

        hidden_states = fuzzing_mask * nvib_hidden_states + (
            (1 - fuzzing_mask) * llm_hidden_state[0]
        )
        attention_out_ = (
            (llm_hidden_state[1], attention_out) if output_attentions else None
        )
        return (
            hidden_states,
            attention_out_,
            kl_g,
            kl_d,
            latent_dict_out,
        )

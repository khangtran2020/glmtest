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
from transformers.cache_utils import Cache
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
        use_cache: bool = False,
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
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
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

        self.use_cache = use_cache
        if use_cache:
            pprint(f"[bold yellow]Using cache in NVIBTransformerLayer[/bold yellow]")
            self.past_key = None
            self.past_value = None
            self.past_pi = None
            self.past_mu = None
            self.past_logvar = None

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

        pprint(
            f"[blue]Entering NVIBTransformerLayer forward with cache mode: {self.use_cache}[/blue]"
        )

        x = src
        if torch.isnan(x).any():
            raise ValueError("NaN detected in the input to NVIBTransformerLayer")

        if latent_dict is not None:
            alpha_skip = latent_dict["alpha"]
        else:
            alpha_skip = None

        out, attention, latent_dict = self._sa_block(
            x,
            src_mask,
            src_key_padding_mask,
            alpha_skip,
        )
        x = x + out
        x = x + self._ff_block(self.norm2(x))

        # Calculate KL divergence
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
        )
        kl_d = self.nvib_layer.kl_dirichlet(
            alpha=latent_dict["alpha"],
            memory_key_padding_mask=latent_dict["memory_key_padding_mask"],
        )
        return x, attention, kl_g, kl_d, latent_dict

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        alpha_skip=None,
        position_emebddings: Optional[Tensor] = None,
    ) -> Tensor:
        # Note query does not include the prior

        latent_dict = self.nvib_layer(x, key_padding_mask, alpha_skip)
        query = x
        key = latent_dict["z"]
        value = latent_dict["z"]

        if self.use_cache:

            (key, pi, mu, logvar) = key  # key is in shape [batch, seq, features]
            (value, pi, mu, logvar) = value  # value is in shape [batch, seq, features]

            if self.past_key is not None and self.past_value is not None:
                key = torch.cat([self.past_key, key], dim=1)
                value = torch.cat([self.past_value, value], dim=1)
                pi = torch.cat([self.past_pi, pi], dim=1)
                mu = torch.cat([self.past_mu, mu], dim=1)
                logvar = torch.cat([self.past_logvar, logvar], dim=1)
            self.past_key = key.clone()
            self.past_value = value.clone()
            self.past_pi = pi.clone()
            self.past_mu = mu.clone()
            self.past_logvar = logvar.clone()

            # debugging shape
            pprint(f"[yellow]Key shape: {key.shape}[/yellow]")
            pprint(f"[cyan]Value shape: {value.shape}[/cyan]")

            key = (
                key,
                pi,
                mu,
                logvar,
            )
            value = (
                value,
                pi,
                mu,
                logvar,
            )

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
        use_cache: bool = False,
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
                use_cache=use_cache,
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
                fuzzing_mask=fuzzing_mask,
            )
        )

        # Check nan in the outcome
        if torch.isnan(nvib_hidden_states).any():
            raise ("NaN detected in nvib_hidden_states")
            pprint(nvib_hidden_states)
        if torch.isnan(attention_out).any():
            raise ("NaN detected in attention_out")
            pprint(attention_out)
        if torch.isnan(kl_g).any():
            raise ("NaN detected in kl_g")
            pprint(kl_g)
        if torch.isnan(kl_d).any():
            raise ("NaN detected in kl_d")
            pprint(kl_d)

        hidden_states = fuzzing_mask * nvib_hidden_states + (
            (1 - fuzzing_mask) * llm_hidden_state[0]
        )

        # Combine the two hidden states
        # debugging
        # Get where fuzzing_mask is 1 and check the outcome at those positions
        # index_fuzz = (fuzzing_mask[0] == 1).nonzero(as_tuple=True)[0]
        # pprint(f"[green]Fuzzing positions: {index_fuzz}[/green]")
        # pprint(f"[green]LLM hidden states: {llm_hidden_state[0]}[/green]")
        # pprint(
        #     f"[green]NVIB hidden states: {nvib_hidden_states[:,index_fuzz,:]}[/green]"
        # )
        # pprint(f"[green]Fuzzed hidden states: {hidden_states[:,index_fuzz,:]}[/green]")

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

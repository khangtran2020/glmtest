import torch
import pytest
from model.layer import GLMFFuzzingLayer
from torch.nn import Module


class DummyLLMLayer(Module):
    def __init__(self, d_model):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(d_model)

    def forward(self, hidden_states, **kwargs):
        return hidden_states


@pytest.mark.parametrize("is_fuzz", [False, True])
def test_glmf_fuzzing_layer_forward(is_fuzz):
    batch_size = 2
    seq_len = 8
    d_model = 16
    nhead = 2
    dim_feedforward = 32
    llm_layer = DummyLLMLayer(d_model)
    layer = GLMFFuzzingLayer(
        d_model=d_model,
        nhead=nhead,
        llm_layer=llm_layer,
        dim_feedforward=dim_feedforward,
        dropout=0.1,
        is_fuzz=is_fuzz,
    )
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    fuzzing_mask = torch.randint(0, 2, (batch_size, seq_len, 1), dtype=torch.float32)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    out = layer(
        hidden_states,
        attention_mask=attention_mask,
        fuzzing_mask=fuzzing_mask,
        position_ids=position_ids,
    )
    if is_fuzz:
        assert isinstance(out, tuple)
        assert out[0].shape == (batch_size, seq_len, d_model)
        assert out[1] is not None
        assert out[2] is not None
        assert out[3] is not None
        assert isinstance(out[4], dict)
    else:
        assert torch.allclose(out, hidden_states, atol=1e-5) or out.shape == hidden_states.shape

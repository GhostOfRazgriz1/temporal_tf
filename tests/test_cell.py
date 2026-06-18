import torch
from temporal_tf.config import Config
from temporal_tf.layers import PredictiveCell

def _cfg(): return Config(image_size=16, patch_size=8, d_model=32, n_heads=4)

def test_cell_output_shapes():
    cfg = _cfg(); cell = PredictiveCell(cfg)
    B = 2; h = cell.init_state(B, "cpu")
    x = torch.randn(B, cfg.n_tokens, cfg.d_model)
    R, P, h2 = cell(x, h)
    for t in (R, P, h2):
        assert t.shape == (B, cfg.n_tokens, cfg.d_model)

def test_state_carries_information():
    cfg = _cfg(); cell = PredictiveCell(cfg); B = 2
    x = torch.randn(B, cfg.n_tokens, cfg.d_model)
    _, _, h1 = cell(x, cell.init_state(B, "cpu"))
    R_a, _, _ = cell(x, h1)
    R_b, _, _ = cell(x, cell.init_state(B, "cpu"))
    assert not torch.allclose(R_a, R_b)  # same input, different state -> different output

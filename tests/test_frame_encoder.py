import torch
from temporal_tf.config import Config
from temporal_tf.layers import FrameEncoder, make_token_mask

def _cfg(): return Config(image_size=16, patch_size=8, d_model=32, n_heads=4)  # n_tokens=4

def test_frame_encoder_output_shape():
    cfg = _cfg(); enc = FrameEncoder(cfg)
    out = enc(torch.randn(2, 1, 16, 16), mask=None)
    assert out.shape == (2, cfg.n_tokens, cfg.d_model)

def test_mask_ratio_and_effect():
    cfg = _cfg(); enc = FrameEncoder(cfg)
    rng = torch.Generator().manual_seed(0)
    mask = make_token_mask(2, cfg.n_tokens, ratio=0.5, rng=rng)
    assert mask.dtype == torch.bool and mask.shape == (2, cfg.n_tokens)
    assert int(mask[0].sum()) == 2  # 50% of 4 tokens
    frame = torch.randn(2, 1, 16, 16)
    assert not torch.allclose(enc(frame, mask=None), enc(frame, mask=mask))

import torch
from temporal_tf.config import Config
from temporal_tf.model import TemporalDepthModel

def _cfg(): return Config(image_size=16, patch_size=8, d_model=32, n_heads=4, n_layers=4, clip_len=6, mask_ratio=0.5)

def test_forward_record_shapes():
    cfg = _cfg(); m = TemporalDepthModel(cfg)
    clip = torch.rand(2, cfg.clip_len, 1, 16, 16)
    out = m(clip, rng=torch.Generator().manual_seed(0))
    assert len(out.features) == cfg.n_layers
    assert len(out.features[0]) == cfg.clip_len
    assert out.surprise.shape == (2, cfg.clip_len, cfg.n_layers)
    assert len(out.pairs) > 0
    for (l, t, P, tgt) in out.pairs:
        assert P.shape == tgt.shape == (2, cfg.n_tokens, cfg.d_model)
        assert not tgt.requires_grad   # target is stop-grad

def test_pair_targets_respect_horizon_A():
    cfg = _cfg(); m = TemporalDepthModel(cfg)   # horizon_mode "A" -> target tick = t (the next feature received)
    out = m(torch.rand(2, cfg.clip_len, 1, 16, 16), rng=torch.Generator().manual_seed(1))
    assert out.pairs, "expected prediction pairs"
    # every pair's target tick (t + h - 1 = t for mode A) is within the clip
    assert all(0 <= t <= cfg.clip_len - 1 for (l, t, P, tgt) in out.pairs)
    # the corrected 1-step horizon now includes the final tick
    assert max(t for (l, t, P, tgt) in out.pairs) == cfg.clip_len - 1

import torch
from temporal_tf.config import Config
from temporal_tf.data import synthetic_digit_bank, generate_clip, generate_batch, _simulate

def _cfg(**kw): return Config(image_size=32, patch_size=8, clip_len=12, **kw)

def test_generate_clip_unchanged_signature():
    cfg = _cfg(); bank = synthetic_digit_bank()
    clip, bounces = generate_clip(torch.Generator().manual_seed(0), bank, cfg)
    assert clip.shape == (cfg.clip_len, 1, 32, 32) and bounces.dtype == torch.bool

def test_teleport_off_by_default():
    cfg = _cfg(teleport_prob=0.0); bank = synthetic_digit_bank()
    _, _, tel = _simulate(torch.Generator().manual_seed(1), bank, cfg, 1, 0.0)
    assert tel.sum() == 0

def test_teleport_always_when_prob_one():
    cfg = _cfg(); bank = synthetic_digit_bank()
    _, bounces, tel = _simulate(torch.Generator().manual_seed(2), bank, cfg, 1, 1.0)
    assert bool(tel.all()) and bounces.sum() == 0   # all teleport, no bounces

def test_generate_batch_shapes_and_events():
    cfg = _cfg(teleport_prob=0.3); bank = synthetic_digit_bank()
    clips, ev = generate_batch(torch.Generator().manual_seed(3), bank, cfg, B=4)
    assert clips.shape == (4, cfg.clip_len, 1, 32, 32)
    assert ev["bounce"].shape == ev["teleport"].shape == (4, cfg.clip_len)
    assert ev["bounce"].dtype == torch.bool

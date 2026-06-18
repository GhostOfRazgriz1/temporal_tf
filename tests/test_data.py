import torch
from temporal_tf.config import default_config
from temporal_tf.data import synthetic_digit_bank, generate_clip

def test_clip_shapes_and_range():
    cfg = default_config()
    bank = synthetic_digit_bank()
    rng = torch.Generator().manual_seed(0)
    clip, bounces = generate_clip(rng, bank, cfg)
    assert clip.shape == (cfg.clip_len, 1, cfg.image_size, cfg.image_size)
    assert bounces.shape == (cfg.clip_len,) and bounces.dtype == torch.bool
    assert clip.min() >= 0.0 and clip.max() <= 1.0

def test_clip_is_deterministic_given_seed():
    cfg = default_config()
    bank = synthetic_digit_bank()
    c1, b1 = generate_clip(torch.Generator().manual_seed(7), bank, cfg)
    c2, b2 = generate_clip(torch.Generator().manual_seed(7), bank, cfg)
    assert torch.allclose(c1, c2) and torch.equal(b1, b2)

def test_a_bounce_is_detected_somewhere():
    cfg = default_config()
    bank = synthetic_digit_bank()
    # high velocity over a long clip guarantees wall hits
    any_bounce = False
    for s in range(10):
        _, b = generate_clip(torch.Generator().manual_seed(s), bank, cfg)
        any_bounce = any_bounce or bool(b.any())
    assert any_bounce

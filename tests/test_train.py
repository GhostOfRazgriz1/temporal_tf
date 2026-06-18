import torch
from temporal_tf.config import Config
from temporal_tf.train import overfit_batch

def test_overfit_reduces_loss():
    cfg = Config(image_size=16, patch_size=8, d_model=32, n_heads=4, n_layers=4,
                 clip_len=6, mask_ratio=0.5, lr=3e-3, seed=0)
    hist = overfit_batch(cfg, steps=40)
    assert hist[-1] < hist[0] * 0.8     # loss drops at least 20% on a fixed batch
    assert all(torch.isfinite(torch.tensor(h)) for h in hist)

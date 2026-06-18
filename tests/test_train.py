import torch
import pytest
from temporal_tf.config import Config
from temporal_tf.train import overfit_batch

def test_overfit_reduces_loss():
    cfg = Config(image_size=16, patch_size=8, d_model=32, n_heads=4, n_layers=4,
                 clip_len=6, mask_ratio=0.5, lr=3e-3, seed=0)
    hist = overfit_batch(cfg, steps=40)
    assert hist[-1] < hist[0] * 0.8     # loss drops at least 20% on a fixed batch
    assert all(torch.isfinite(torch.tensor(h)) for h in hist)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_train_step_runs_on_cuda():
    from temporal_tf.model import TemporalDepthModel
    from temporal_tf.train import train_step
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=3, clip_len=4, mask_ratio=0.5)
    model = TemporalDepthModel(cfg)
    model.online.to("cuda"); model.target.module.to("cuda")
    opt = torch.optim.Adam(model.online.parameters(), lr=1e-3)
    from temporal_tf.data import synthetic_digit_bank, generate_clip
    clip, _ = generate_clip(torch.Generator().manual_seed(0), synthetic_digit_bank(size=cfg.image_size // 2), cfg)
    parts = train_step(model, clip.unsqueeze(0).to("cuda"), opt, cfg, torch.Generator().manual_seed(0))
    assert "total" in parts

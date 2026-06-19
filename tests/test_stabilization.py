import math
import torch
from temporal_tf.config import Config
from temporal_tf.train import _lr_lambda, train

def test_lr_lambda_warmup_then_cosine():
    cfg = Config(warmup_steps=10, lr_schedule="cosine", lr=1e-3)
    assert abs(_lr_lambda(0, cfg, 100) - 0.1) < 1e-6          # first warmup step
    assert abs(_lr_lambda(9, cfg, 100) - 1.0) < 1e-6          # end of warmup
    mid = _lr_lambda(55, cfg, 100)                            # halfway through cosine
    assert 0.3 < mid < 0.7
    assert _lr_lambda(100, cfg, 100) < 1e-3                   # decays to ~0 at the end

def test_lr_lambda_none_is_constant_after_warmup():
    cfg = Config(warmup_steps=0, lr_schedule="none")
    assert _lr_lambda(0, cfg, 100) == 1.0 and _lr_lambda(50, cfg, 100) == 1.0

def test_train_runs_stabilized():
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=3,
                 clip_len=4, mask_ratio=0.5, batch_size=2, grad_clip=1.0,
                 warmup_steps=2, lr_schedule="cosine", seed=0)
    model = train(cfg, n_steps=5)
    for p in model.online.parameters():
        assert torch.isfinite(p).all()

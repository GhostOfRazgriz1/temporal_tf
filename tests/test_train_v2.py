import torch
from temporal_tf.config import Config
from temporal_tf.train import train

def _cfg(): return Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=3,
                          clip_len=4, mask_ratio=0.5, batch_size=3, seed=0)

def test_train_runs_batched_and_calls_on_eval():
    cfg = _cfg()
    seen = []
    model = train(cfg, n_steps=4, on_eval=lambda m, s, p: seen.append(s), eval_every=2)
    assert seen == [0, 2]                       # called at steps 0 and 2
    assert next(model.online.parameters()).requires_grad

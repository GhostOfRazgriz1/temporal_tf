import torch
from temporal_tf.config import Config
from temporal_tf.run_prototype import run

def test_end_to_end_smoke():
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=4,
                 clip_len=6, mask_ratio=0.5, lr=3e-3, seed=0)
    report = run(cfg=cfg, n_steps=20, use_mnist=False)
    assert "auc_per_layer" in report and len(report["auc_per_layer"]) == cfg.n_layers - 1
    assert "collapse_std_per_layer" in report
    for v in report["auc_per_layer"]:
        assert 0.0 <= v <= 1.0

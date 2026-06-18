import torch
from temporal_tf.config import Config
from temporal_tf.run_prototype import run

def test_end_to_end_smoke():
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=4,
                 clip_len=6, mask_ratio=0.5, lr=3e-3, seed=0)
    report = run(cfg=cfg, n_steps=20, use_mnist=False)
    assert "auc_bounce_per_layer" in report and len(report["auc_bounce_per_layer"]) == cfg.n_layers - 1
    assert "collapse_std_per_layer" in report
    assert "cross_layer_surprise_corr" in report
    for v in report["auc_bounce_per_layer"]:
        assert 0.0 <= v <= 1.0

def test_end_to_end_smoke_v2():
    from temporal_tf.config import Config
    from temporal_tf.run_prototype import run
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=4,
                 clip_len=6, mask_ratio=0.5, lr=3e-3, seed=0,
                 batch_size=2, teleport_prob=0.3, n_mask_draws=2, n_eval_clips=3)
    report = run(cfg=cfg, n_steps=6, use_mnist=False, track_every=3)
    for k in ("auc_bounce_per_layer", "auc_teleport_per_layer", "collapse_std_per_layer",
              "cross_layer_surprise_corr", "training_curve"):
        assert k in report
    assert len(report["auc_bounce_per_layer"]) == cfg.n_layers - 1
    assert len(report["auc_teleport_per_layer"]) == cfg.n_layers - 1
    assert len(report["training_curve"]) == 2          # steps 0 and 3
    for v in report["auc_teleport_per_layer"]:
        assert 0.0 <= v <= 1.0

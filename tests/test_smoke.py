import torch
from temporal_tf.config import Config
from temporal_tf.run_prototype import run


def test_smoke_tracks_probe_over_training():
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=4,
                 clip_len=6, mask_ratio=0.5, lr=3e-3, seed=0, batch_size=2,
                 teleport_prob=0.3, n_mask_draws=1, n_eval_clips=5, surprise_topk=2)
    report = run(cfg=cfg, n_steps=6, use_mnist=False, track_every=3)
    assert "best_error_probe_teleport" in report
    assert set(report["best_error_probe_teleport"]) == {"step", "auc"}
    entry = report["loss_curve"][0]
    assert "error_probe_auc" in entry and "teleport" in entry["error_probe_auc"]


def test_end_to_end_smoke_v3():
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=4,
                 clip_len=6, mask_ratio=0.5, lr=3e-3, seed=0,
                 batch_size=2, teleport_prob=0.3, n_mask_draws=1, n_eval_clips=4, surprise_topk=2)
    report = run(cfg=cfg, n_steps=6, use_mnist=False, track_every=3)
    for k in ("loss_curve", "raw_pooled_auc", "error_probe_auc", "feature_probe_auc", "collapse_std_per_layer"):
        assert k in report
    assert len(report["loss_curve"]) == 2 and "pred" in report["loss_curve"][0]
    for ev in ("bounce", "teleport"):
        assert set(report["raw_pooled_auc"][ev]) == {"mean", "max", "topk"}
        assert len(report["raw_pooled_auc"][ev]["max"]) == cfg.n_layers - 1
        assert 0.0 <= report["error_probe_auc"][ev] <= 1.0
        assert 0.0 <= report["feature_probe_auc"][ev] <= 1.0

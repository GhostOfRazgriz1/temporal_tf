from temporal_tf.config import Config
from temporal_tf.run_prototype import run_seeds


def test_run_seeds_aggregates():
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=3,
                 clip_len=6, mask_ratio=0.5, batch_size=2, teleport_prob=0.3,
                 n_mask_draws=1, n_eval_clips=4, surprise_topk=2, lr=3e-3)
    agg = run_seeds(cfg, n_steps=4, seeds=[0, 1], use_mnist=False)
    m = agg["final_error_probe_teleport"]
    assert set(m) == {"mean", "std", "values"} and len(m["values"]) == 2
    assert 0.0 <= m["mean"] <= 1.0

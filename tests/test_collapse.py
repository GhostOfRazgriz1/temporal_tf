import torch
from temporal_tf.eval.collapse import representation_stats, is_collapsed

def test_detects_collapse_vs_healthy():
    collapsed = torch.zeros(64, 32)
    healthy = torch.randn(64, 32)
    assert is_collapsed(collapsed) and not is_collapsed(healthy)
    assert representation_stats(healthy)["eff_rank"] > representation_stats(collapsed)["eff_rank"]

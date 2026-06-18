import torch
from temporal_tf.losses import predictive_loss, vicreg

def test_predictive_loss_zero_when_equal():
    x = torch.randn(2, 4, 8)
    pairs = [(1, 0, x, x.detach())]
    assert predictive_loss(pairs).item() < 1e-6

def test_vicreg_penalizes_collapse():
    collapsed = torch.zeros(16, 8)            # no variance -> high variance loss
    spread = torch.randn(16, 8) * 2.0
    assert vicreg(collapsed, 25.0, 1.0) > vicreg(spread, 25.0, 1.0)

import torch
from temporal_tf.eval.timescale import cross_layer_correlation
from temporal_tf.eval.probe import linear_probe

def test_correlation_identity_diag():
    T, L = 30, 3
    s = torch.randn(T, L)
    c = cross_layer_correlation(s)
    assert c.shape == (L, L)
    assert torch.allclose(c.diag(), torch.ones(L), atol=1e-5)

def test_probe_learns_separable_features():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(50, 8, generator=g) + 3.0
    b = torch.randn(50, 8, generator=g) - 3.0
    X = torch.cat([a, b]); y = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    assert linear_probe(X, y) > 0.9

import torch
from temporal_tf.config import Config
from temporal_tf.model import TemporalDepthModel
from temporal_tf.eval.event_probe import surprise_maps, pooled_auc, per_tick_error_features, event_probe

def _cfg(): return Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=4,
                          clip_len=6, mask_ratio=0.5, surprise_topk=2)

def test_surprise_maps_shapes_and_layer0_nan():
    cfg = _cfg(); m = TemporalDepthModel(cfg)
    out = m(torch.rand(2, cfg.clip_len, 1, 16, 16), rng=torch.Generator().manual_seed(0))
    maps = surprise_maps(out, cfg.n_layers, cfg.surprise_topk)
    for r in ("mean", "max", "topk"):
        assert maps[r].shape == (2, cfg.clip_len, cfg.n_layers)
        assert torch.isnan(maps[r][:, :, 0]).all()                 # layer 0 never predicted
    # max >= mean wherever both defined
    defined = ~torch.isnan(maps["mean"])
    assert (maps["max"][defined] >= maps["mean"][defined] - 1e-6).all()

def test_pooled_auc():
    labels = torch.tensor([False, False, True, True])
    assert pooled_auc(torch.tensor([0.1, 0.2, 0.9, 0.8]), labels) > 0.9
    assert pooled_auc(torch.ones(4), labels) == 0.5                # constant
    assert pooled_auc(torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.tensor([True, True, True, True])) == 0.5  # degenerate

def test_per_tick_error_features_shape():
    cfg = _cfg(); m = TemporalDepthModel(cfg)
    out = m(torch.rand(2, cfg.clip_len, 1, 16, 16), rng=torch.Generator().manual_seed(0))
    feats = per_tick_error_features(out, cfg.n_layers, cfg.surprise_topk)
    assert feats.shape == (2, cfg.clip_len, (cfg.n_layers - 1) * 3)

def test_event_probe_separates():
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(60, 6, generator=g) + 2.5
    neg = torch.randn(60, 6, generator=g) - 2.5
    X = torch.cat([pos, neg]); y = torch.cat([torch.ones(60), torch.zeros(60)]).bool()
    assert event_probe(X, y) > 0.9
    assert abs(event_probe(torch.randn(120, 6, generator=g), y) - 0.5) < 0.2  # random ~ chance

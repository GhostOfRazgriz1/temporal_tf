import torch
from temporal_tf.eval.surprise_eval import localization_auc

def test_auc_high_when_surprise_tracks_events():
    T = 20
    events = torch.zeros(T, dtype=torch.bool); events[[5, 12, 17]] = True
    surprise = torch.rand(T) * 0.1
    surprise[events] += 1.0                       # spikes at events
    assert localization_auc(surprise, events) > 0.9

def test_auc_chance_when_flat():
    T = 20
    events = torch.zeros(T, dtype=torch.bool); events[[3, 9]] = True
    assert abs(localization_auc(torch.ones(T), events) - 0.5) < 1e-6

import torch
from sklearn.metrics import roc_auc_score

def localization_auc(surprise_layer, events):
    s = surprise_layer.detach().cpu()
    e = events.cpu().bool()
    valid = ~torch.isnan(s)
    s, e = s[valid], e[valid]
    if e.sum() == 0 or e.sum() == len(e):
        return 0.5
    if torch.allclose(s, s[0].expand_as(s)):
        return 0.5
    return float(roc_auc_score(e.numpy(), s.numpy()))

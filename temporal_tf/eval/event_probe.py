import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def _token_err(P, target):
    # P, target: (B, N, d) -> per-token RMS error (B, N)
    return (target - P).pow(2).mean(-1).sqrt()

def surprise_maps(out, n_layers, topk):
    # find B, T from any pair (fallback to surprise tensor shape)
    B, T = out.surprise.shape[0], out.surprise.shape[1]
    maps = {r: torch.full((B, T, n_layers), float("nan")) for r in ("mean", "max", "topk")}
    for (li, t, P, target) in out.pairs:
        e = _token_err(P, target)                      # (B, N)
        k = min(topk, e.shape[-1])
        maps["mean"][:, t, li] = e.mean(-1)
        maps["max"][:, t, li] = e.max(-1).values
        maps["topk"][:, t, li] = e.topk(k, dim=-1).values.mean(-1)
    return maps

def pooled_auc(scores, labels):
    s = scores.detach().cpu().flatten()
    y = labels.detach().cpu().bool().flatten()
    valid = ~torch.isnan(s)
    s, y = s[valid], y[valid]
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    if torch.allclose(s, s[0].expand_as(s)):
        return 0.5
    return float(roc_auc_score(y.numpy(), s.numpy()))

def per_tick_error_features(out, n_layers, topk):
    B, T = out.surprise.shape[0], out.surprise.shape[1]
    n_pred = n_layers - 1
    feats = torch.full((B, T, n_pred * 3), float("nan"))
    for (li, t, P, target) in out.pairs:
        e = _token_err(P, target)                      # (B, N)
        k = min(topk, e.shape[-1])
        base = (li - 1) * 3
        feats[:, t, base + 0] = e.mean(-1)
        feats[:, t, base + 1] = e.max(-1).values
        feats[:, t, base + 2] = e.topk(k, dim=-1).values.mean(-1)
    return feats

def event_probe(features, labels, test_size=0.3, seed=0):
    X = features.detach().cpu().reshape(-1, features.shape[-1]).numpy()
    y = labels.detach().cpu().bool().reshape(-1).numpy()
    import numpy as np
    keep = ~np.isnan(X).any(1)
    X, y = X[keep], y[keep]
    if y.sum() < 2 or (~y).sum() < 2:
        return 0.5
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    if ytr.sum() == 0 or yte.sum() == 0:
        return 0.5
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))

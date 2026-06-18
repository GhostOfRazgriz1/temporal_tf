import torch
import torch.nn.functional as F

def predictive_loss(pairs):
    if not pairs:
        return torch.tensor(0.0)
    return torch.stack([F.mse_loss(P, tgt) for (_, _, P, tgt) in pairs]).mean()

def vicreg(features_flat, var_coef, cov_coef, eps=1e-4):
    x = features_flat - features_flat.mean(0, keepdim=True)
    std = torch.sqrt(x.var(0) + eps)
    var_term = torch.relu(1.0 - std).mean()
    M, d = x.shape
    cov = (x.T @ x) / (M - 1)
    cov_term = (cov.pow(2).sum() - cov.diag().pow(2).sum()) / d
    return var_coef * var_term + cov_coef * cov_term

def total_loss(out, cfg):
    pred = predictive_loss(out.pairs)
    feats = []
    for li in range(1, cfg.n_layers):
        for t in range(len(out.features[li])):
            f = out.features[li][t]
            feats.append(f.reshape(-1, f.shape[-1]))
    flat = torch.cat(feats, 0)
    x = flat - flat.mean(0, keepdim=True)
    std = torch.sqrt(x.var(0) + 1e-4)
    var = torch.relu(1.0 - std).mean()
    M, d = x.shape
    cov = (x.T @ x) / (M - 1)
    cov_t = (cov.pow(2).sum() - cov.diag().pow(2).sum()) / d
    loss = cfg.pred_coef * pred + cfg.var_coef * var + cfg.cov_coef * cov_t
    return loss, {"pred": float(pred), "var": float(var), "cov": float(cov_t)}

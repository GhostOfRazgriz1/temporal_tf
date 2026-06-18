import torch

def cross_layer_correlation(surprise):
    s = surprise.detach().cpu()
    valid = ~torch.isnan(s).any(1)
    s = s[valid]
    s = s - s.mean(0, keepdim=True)
    std = s.std(0, keepdim=True) + 1e-9
    s = s / std
    return (s.T @ s) / (s.shape[0] - 1)

import torch

def representation_stats(R):
    x = R.reshape(-1, R.shape[-1])
    std = float(x.std(0).mean())
    xc = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(xc)
    p = s / (s.sum() + 1e-9)
    log_p = torch.log(p.clamp(min=1e-9))
    eff_rank = float(torch.exp(-(p * log_p).sum()))
    return {"std": std, "eff_rank": eff_rank}

def is_collapsed(R, std_thresh=1e-2):
    return representation_stats(R)["std"] < std_thresh

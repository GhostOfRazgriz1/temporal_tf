from dataclasses import dataclass
import torch
import torch.nn as nn
from .layers import FrameEncoder, PredictiveCell, make_token_mask
from .ema import EMATarget

@dataclass
class Output:
    features: list           # [layer][tick] -> (B,N,d)
    pairs: list              # (layer, tick, P, target_sg)
    surprise: torch.Tensor   # (B,T,n_layers)

class _OnlineStack(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = FrameEncoder(cfg)
        self.cells = nn.ModuleList([PredictiveCell(cfg) for _ in range(cfg.n_layers - 1)])

class TemporalDepthModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.online = _OnlineStack(cfg)
        self.target = EMATarget(self.online)   # target.module is the EMA twin of the online stack

    def horizon(self, l):
        return 1 if self.cfg.horizon_mode == "A" else l

    def update_target(self):
        self.target.update(self.online, self.cfg.ema_momentum)

    def _run_stack(self, stack, clip, masks):
        """Returns features[layer][tick] and predictions[layer][tick] (pred None for layer 0)."""
        cfg = self.cfg
        B, T = clip.shape[0], clip.shape[1]
        n = cfg.n_layers
        feats = [[None] * T for _ in range(n)]
        preds = [[None] * T for _ in range(n)]
        states = [cell.init_state(B, clip.device) for cell in stack.cells]
        prev_R = [None] * n
        for t in range(T):
            curr_R = [None] * n
            curr_R[0] = stack.encoder(clip[:, t], mask=masks[t])
            for li, cell in enumerate(stack.cells, start=1):
                x_in = prev_R[li - 1] if prev_R[li - 1] is not None \
                    else torch.zeros(B, cfg.n_tokens, cfg.d_model, device=clip.device)
                R, P, states[li - 1] = cell(x_in, states[li - 1])
                curr_R[li], preds[li][t] = R, P
            for li in range(n):
                feats[li][t] = curr_R[li]
            prev_R = curr_R
        return feats, preds

    def forward(self, clip, rng=None):
        cfg = self.cfg
        B, T = clip.shape[0], clip.shape[1]
        rng = rng or torch.Generator(device="cpu").manual_seed(cfg.seed)
        masks = [make_token_mask(B, cfg.n_tokens, cfg.mask_ratio, rng).to(clip.device) for _ in range(T)]

        feats, preds = self._run_stack(self.online, clip, masks)
        with torch.no_grad():
            tgt_feats, _ = self._run_stack(self.target.module, clip, masks)

        pairs = []
        surprise = torch.full((B, T, cfg.n_layers), float("nan"), device=clip.device)
        for li in range(1, cfg.n_layers):
            h = self.horizon(li)
            for t in range(T):
                tgt_t = t + h - 1
                if preds[li][t] is None or tgt_t >= T:
                    continue
                target = tgt_feats[li - 1][tgt_t].detach()   # stop-grad EMA target
                P = preds[li][t]
                pairs.append((li, t, P, target))
                surprise[:, t, li] = (target - P).pow(2).mean(-1).sqrt().mean(-1)
        return Output(features=feats, pairs=pairs, surprise=surprise)

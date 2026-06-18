import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

def make_token_mask(B, n_tokens, ratio, rng):
    k = int(round(n_tokens * ratio))
    mask = torch.zeros(B, n_tokens, dtype=torch.bool)
    for b in range(B):
        idx = torch.randperm(n_tokens, generator=rng)[:k]
        mask[b, idx] = True
    return mask

class FrameEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.in_channels * cfg.patch_size ** 2
        self.proj = nn.Linear(patch_dim, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.n_tokens, cfg.d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.block = TransformerBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, frame):
        p = self.cfg.patch_size
        B, C, H, W = frame.shape
        x = frame.unfold(2, p, p).unfold(3, p, p)          # B,C,H/p,W/p,p,p
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)
        return x

    def forward(self, frame, mask=None):
        x = self.proj(self.patchify(frame)) + self.pos_embed
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token, x)
        return self.block(x)

class PredictiveCell(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.n_tokens = cfg.n_tokens
        self.d_model = d
        self.in_norm = nn.LayerNorm(d)
        self.block = TransformerBlock(d, cfg.n_heads, cfg.mlp_ratio)   # spatial mixing
        self.gru = nn.GRUCell(d, d)                                    # temporal recurrence (per token)
        self.refine = nn.Linear(d, d)                                  # -> refined feature R
        self.predict = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))  # -> prediction P

    def init_state(self, B, device):
        return torch.zeros(B, self.n_tokens, self.d_model, device=device)

    def forward(self, x_in, h_prev):
        ctx = self.block(self.in_norm(x_in))                  # spatial self-attention over incoming feature
        B, N, d = ctx.shape
        h_new = self.gru(ctx.reshape(B * N, d), h_prev.reshape(B * N, d)).reshape(B, N, d)
        R = self.refine(h_new)
        P = self.predict(h_new)                               # firewalled: depends only on state, not the target
        return R, P, h_new

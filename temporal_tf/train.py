import torch
from .config import Config
from .model import TemporalDepthModel
from .losses import total_loss
from .data import synthetic_digit_bank, generate_clip

def train_step(model, clip, optimizer, cfg, rng):
    model.train()
    out = model(clip, rng=rng)
    loss, parts = total_loss(out, cfg)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.update_target()
    parts["total"] = float(loss)
    return parts

def overfit_batch(cfg: Config, steps: int, clip=None):
    torch.manual_seed(cfg.seed)
    model = TemporalDepthModel(cfg)
    opt = torch.optim.Adam(model.online.parameters(), lr=cfg.lr)
    if clip is None:
        bank = synthetic_digit_bank(size=cfg.image_size // 2)
        clip, _ = generate_clip(torch.Generator().manual_seed(cfg.seed), bank, cfg)
        clip = clip.unsqueeze(0)                       # B=1
    rng = torch.Generator().manual_seed(cfg.seed)
    hist = []
    for _ in range(steps):
        hist.append(train_step(model, clip, opt, cfg, rng)["total"])
    return hist

def train(cfg: Config, n_steps: int, digit_bank=None):
    torch.manual_seed(cfg.seed)
    bank = digit_bank if digit_bank is not None else synthetic_digit_bank(size=cfg.image_size // 2)
    model = TemporalDepthModel(cfg)
    opt = torch.optim.Adam(model.online.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = torch.Generator().manual_seed(cfg.seed + 1)
    for step in range(n_steps):
        clip, _ = generate_clip(gen, bank, cfg)
        parts = train_step(model, clip.unsqueeze(0), opt, cfg, rng)
        if step % 50 == 0:
            print(f"step {step}: {parts}")
    return model

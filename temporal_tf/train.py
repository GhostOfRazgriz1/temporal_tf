import torch
from .config import Config
from .model import TemporalDepthModel
from .losses import total_loss
from .data import synthetic_digit_bank, generate_clip, generate_batch

def train_step(model, clip, optimizer, cfg, rng):
    model.train()
    out = model(clip, rng=rng)
    loss, parts = total_loss(out, cfg)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.update_target()
    parts["total"] = loss.detach().item()
    return parts

def overfit_batch(cfg: Config, steps: int, clip=None, device=None):
    torch.manual_seed(cfg.seed)
    model = TemporalDepthModel(cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.online.to(device)
    model.target.module.to(device)   # EMATarget.module is a plain attr; model.to() would NOT move it
    opt = torch.optim.Adam(model.online.parameters(), lr=cfg.lr)
    if clip is None:
        bank = synthetic_digit_bank(size=cfg.image_size // 2)
        clip, _ = generate_clip(torch.Generator().manual_seed(cfg.seed), bank, cfg)
        clip = clip.unsqueeze(0)                       # B=1
    clip = clip.to(device)
    rng = torch.Generator().manual_seed(cfg.seed)
    hist = []
    for _ in range(steps):
        hist.append(train_step(model, clip, opt, cfg, rng)["total"])
    return hist

def train(cfg: Config, n_steps: int, digit_bank=None, device=None, on_eval=None, eval_every=0):
    torch.manual_seed(cfg.seed)
    bank = digit_bank if digit_bank is not None else synthetic_digit_bank(size=cfg.image_size // 2)
    model = TemporalDepthModel(cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.online.to(device)
    model.target.module.to(device)   # EMATarget is a plain attr; model.to() would not reach it
    opt = torch.optim.Adam(model.online.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = torch.Generator().manual_seed(cfg.seed + 1)
    for step in range(n_steps):
        clips, _ = generate_batch(gen, bank, cfg, cfg.batch_size)
        parts = train_step(model, clips.to(device), opt, cfg, rng)
        if on_eval is not None and eval_every > 0 and step % eval_every == 0:
            on_eval(model, step)
        if step % 50 == 0:
            print(f"step {step}: {parts}")
    return model

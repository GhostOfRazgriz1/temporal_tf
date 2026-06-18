import torch
from .config import Config

def synthetic_digit_bank(k: int = 4, size: int = 28) -> torch.Tensor:
    bank = torch.zeros(k, size, size)
    for i in range(k):
        lo, hi = size // 4, size - size // 4
        bank[i, lo:hi, lo:hi] = 0.4 + 0.6 * (i + 1) / k  # filled square, varied intensity
    return bank

def load_mnist_digit_bank(root: str) -> torch.Tensor:
    from torchvision.datasets import MNIST
    ds = MNIST(root, train=True, download=True)
    imgs = ds.data.float() / 255.0          # (60000,28,28)
    return imgs

def generate_clip(rng: torch.Generator, digit_bank: torch.Tensor, cfg: Config,
                  n_digits: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    H = W = cfg.image_size
    ds = digit_bank.shape[-1]
    T = cfg.clip_len
    clip = torch.zeros(T, 1, H, W)
    bounces = torch.zeros(T, dtype=torch.bool)
    idx = torch.randint(0, digit_bank.shape[0], (n_digits,), generator=rng)
    pos = torch.rand(n_digits, 2, generator=rng) * torch.tensor([H - ds, W - ds]).float()
    vel = (torch.rand(n_digits, 2, generator=rng) * 2 - 1) * 4.0  # up to 4 px/tick
    for t in range(T):
        for d in range(n_digits):
            for ax, limit in enumerate((H - ds, W - ds)):
                pos[d, ax] += vel[d, ax]
                if pos[d, ax] < 0 or pos[d, ax] > limit:
                    vel[d, ax] = -vel[d, ax]
                    pos[d, ax] = pos[d, ax].clamp(0, limit)
                    bounces[t] = True
            y, x = int(pos[d, 0]), int(pos[d, 1])
            patch = digit_bank[idx[d]]
            clip[t, 0, y:y + ds, x:x + ds] = torch.maximum(clip[t, 0, y:y + ds, x:x + ds], patch)
    return clip.clamp(0, 1), bounces

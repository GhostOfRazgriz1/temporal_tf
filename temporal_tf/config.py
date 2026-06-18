from dataclasses import dataclass

@dataclass
class Config:
    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 1
    d_model: int = 128
    n_heads: int = 4
    mlp_ratio: float = 2.0
    n_layers: int = 6          # L0 (encoder) + (n_layers-1) predictive cells
    clip_len: int = 20
    mask_ratio: float = 0.75
    horizon_mode: str = "A"    # "A": horizon 1 for all; "B": layer l predicts l ticks ahead
    ema_momentum: float = 0.99
    var_coef: float = 25.0
    cov_coef: float = 1.0
    pred_coef: float = 1.0
    tbptt: int = 5
    lr: float = 1e-3
    seed: int = 0
    teleport_prob: float = 0.0
    batch_size: int = 1
    n_mask_draws: int = 1
    n_eval_clips: int = 8

    @property
    def n_tokens(self) -> int:
        return (self.image_size // self.patch_size) ** 2

def default_config() -> Config:
    return Config()

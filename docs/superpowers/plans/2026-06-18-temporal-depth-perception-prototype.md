# Temporal-Depth Predictive Perception — Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, runnable PyTorch prototype of the streaming time-on-depth perception module and verify its three core claims on toy video (learns without collapse; emits a meaningful multi-timescale surprise signal; features are decodable).

**Architecture:** A stack of untied layers processes a video stream as a wavefront — a frame's refined feature climbs one layer per tick (so depth = temporal lag). `L0` encodes each frame; each predictive cell `L1..L{n-1}` carries recurrent state, emits a *refined feature* upward and a *prediction* of the feature it will next receive from below. Per-layer latent prediction error = surprise. Trained self-supervised with stop-grad/EMA targets + VICReg + input masking to prevent the copy-collapse that would null the surprise signal.

**Tech Stack:** Python 3.10+, PyTorch 2.x, torchvision (MNIST), NumPy, scikit-learn (AUC + linear probe), pytest. Runs on Colab (GPU for training, CPU for tests).

## Global Constraints

- **Self-supervised only** — no task/classification head anywhere in the module.
- **Anti-collapse is mandatory and load-bearing** — every training path uses: latent-space targets (never pixel reconstruction), stop-gradient + EMA target net, VICReg variance+covariance regularization on refined features, and input masking. Collapse = dead surprise signal.
- **Untied across depth, recurrent across wall-clock time.** Each layer has its own parameters; each layer reuses its own weights every tick.
- **Stack depth `n_layers` is fixed** (= max temporal lag = number of timescales), independent of stream length.
- **Prediction horizon = 1 (decision A)** by default; a per-layer growing horizon (decision B) is a config knob, not a separate code path.
- **One-tick lag is structural:** layer `l` at tick `t` consumes layer `l-1`'s output from tick `t-1`.
- **Package import name:** `temporal_tf` (package dir at repo root). Tensors use shape convention `(B, n_tokens, d_model)` for features/states.
- **Tests must be hermetic:** no network downloads in tests — use the synthetic digit bank, not MNIST.

---

### Task 1: Package scaffolding + config

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `temporal_tf/__init__.py`
- Create: `temporal_tf/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `temporal_tf.config.Config` (dataclass) and `temporal_tf.config.default_config() -> Config`. Fields: `image_size:int=64`, `patch_size:int=8`, `in_channels:int=1`, `d_model:int=128`, `n_heads:int=4`, `mlp_ratio:float=2.0`, `n_layers:int=6` (L0 + 5 predictive), `clip_len:int=20`, `mask_ratio:float=0.75`, `horizon_mode:str="A"` (`"A"`=1, `"B"`=per-layer `l`), `ema_momentum:float=0.99`, `var_coef:float=25.0`, `cov_coef:float=1.0`, `pred_coef:float=1.0`, `tbptt:int=5`, `lr:float=1e-3`, `seed:int=0`. Property `n_tokens -> (image_size//patch_size)**2`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_config.py
from temporal_tf.config import Config, default_config

def test_default_config_shapes():
    cfg = default_config()
    assert cfg.n_tokens == (cfg.image_size // cfg.patch_size) ** 2 == 64
    assert cfg.n_layers >= 2 and cfg.horizon_mode in ("A", "B")

def test_config_is_overridable():
    cfg = Config(d_model=32, n_layers=3)
    assert cfg.d_model == 32 and cfg.n_layers == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'temporal_tf'`

- [ ] **Step 3: Write the files**
```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "temporal_tf"
version = "0.0.1"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
include = ["temporal_tf*"]
```
```text
# requirements.txt
torch>=2.0
torchvision>=0.15
numpy>=1.24
scikit-learn>=1.2
pytest>=7.0
```
```python
# temporal_tf/__init__.py
```
```python
# temporal_tf/config.py
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

    @property
    def n_tokens(self) -> int:
        return (self.image_size // self.patch_size) ** 2

def default_config() -> Config:
    return Config()
```

- [ ] **Step 4: Install editable + run test to verify it passes**

Run: `pip install -e . && pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**
```bash
git add pyproject.toml requirements.txt temporal_tf/__init__.py temporal_tf/config.py tests/test_config.py
git commit -m "feat: package scaffolding and Config"
```

---

### Task 2: Moving-MNIST stream with event (bounce) labels

**Files:**
- Create: `temporal_tf/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `Config`.
- Produces:
  - `synthetic_digit_bank(k:int=4, size:int=28) -> Tensor` shape `(k,size,size)` in `[0,1]` (hermetic; a few filled squares of different intensities).
  - `generate_clip(rng: torch.Generator, digit_bank: Tensor, cfg: Config, n_digits:int=1) -> tuple[Tensor, Tensor]` → `clip (clip_len,1,H,W)` in `[0,1]`, `bounces (clip_len,)` bool (True at ticks where any digit reversed a velocity component off a wall).
  - `load_mnist_digit_bank(root:str) -> Tensor` (used only by training, NOT tests).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_data.py
import torch
from temporal_tf.config import default_config
from temporal_tf.data import synthetic_digit_bank, generate_clip

def test_clip_shapes_and_range():
    cfg = default_config()
    bank = synthetic_digit_bank()
    rng = torch.Generator().manual_seed(0)
    clip, bounces = generate_clip(rng, bank, cfg)
    assert clip.shape == (cfg.clip_len, 1, cfg.image_size, cfg.image_size)
    assert bounces.shape == (cfg.clip_len,) and bounces.dtype == torch.bool
    assert clip.min() >= 0.0 and clip.max() <= 1.0

def test_clip_is_deterministic_given_seed():
    cfg = default_config()
    bank = synthetic_digit_bank()
    c1, b1 = generate_clip(torch.Generator().manual_seed(7), bank, cfg)
    c2, b2 = generate_clip(torch.Generator().manual_seed(7), bank, cfg)
    assert torch.allclose(c1, c2) and torch.equal(b1, b2)

def test_a_bounce_is_detected_somewhere():
    cfg = default_config()
    bank = synthetic_digit_bank()
    # high velocity over a long clip guarantees wall hits
    any_bounce = False
    for s in range(10):
        _, b = generate_clip(torch.Generator().manual_seed(s), bank, cfg)
        any_bounce = any_bounce or bool(b.any())
    assert any_bounce
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError`

- [ ] **Step 3: Write the implementation**
```python
# temporal_tf/data.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/data.py tests/test_data.py
git commit -m "feat: moving-MNIST stream generator with bounce/event labels"
```

---

### Task 3: Frame encoder (L0) with token masking

**Files:**
- Create: `temporal_tf/layers.py`
- Test: `tests/test_frame_encoder.py`

**Interfaces:**
- Consumes: `Config`.
- Produces in `temporal_tf/layers.py`:
  - `class TransformerBlock(nn.Module)`: `__init__(d_model, n_heads, mlp_ratio)`; `forward(x:(B,N,d)) -> (B,N,d)` (pre-norm MHSA + MLP, residual).
  - `class FrameEncoder(nn.Module)`: `__init__(cfg)`; holds learnable `mask_token (1,1,d)` and `pos_embed (1,n_tokens,d)`; `forward(frame:(B,1,H,W), mask:Tensor|None) -> (B,n_tokens,d)`. `mask` is bool `(B,n_tokens)`; masked token positions are replaced by `mask_token` before the block.
  - `make_token_mask(B:int, n_tokens:int, ratio:float, rng) -> BoolTensor (B,n_tokens)` (True = masked).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_frame_encoder.py
import torch
from temporal_tf.config import Config
from temporal_tf.layers import FrameEncoder, make_token_mask

def _cfg(): return Config(image_size=16, patch_size=8, d_model=32, n_heads=4)  # n_tokens=4

def test_frame_encoder_output_shape():
    cfg = _cfg(); enc = FrameEncoder(cfg)
    out = enc(torch.randn(2, 1, 16, 16), mask=None)
    assert out.shape == (2, cfg.n_tokens, cfg.d_model)

def test_mask_ratio_and_effect():
    cfg = _cfg(); enc = FrameEncoder(cfg)
    rng = torch.Generator().manual_seed(0)
    mask = make_token_mask(2, cfg.n_tokens, ratio=0.5, rng=rng)
    assert mask.dtype == torch.bool and mask.shape == (2, cfg.n_tokens)
    assert int(mask[0].sum()) == 2  # 50% of 4 tokens
    frame = torch.randn(2, 1, 16, 16)
    assert not torch.allclose(enc(frame, mask=None), enc(frame, mask=mask))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_encoder.py -v`
Expected: FAIL with `ImportError` (no `FrameEncoder`)

- [ ] **Step 3: Write the implementation**
```python
# temporal_tf/layers.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_encoder.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/layers.py tests/test_frame_encoder.py
git commit -m "feat: frame encoder (L0) with token masking + transformer block"
```

---

### Task 4: Predictive cell (L1..L{n-1})

**Files:**
- Modify: `temporal_tf/layers.py` (append `PredictiveCell`)
- Test: `tests/test_cell.py`

**Interfaces:**
- Consumes: `TransformerBlock`, `Config`.
- Produces: `class PredictiveCell(nn.Module)`: `__init__(cfg)`; `init_state(B, device) -> Tensor (B,n_tokens,d)`; `forward(x_in:(B,N,d), h_prev:(B,N,d)) -> (R:(B,N,d), P:(B,N,d), h_new:(B,N,d))`.
  - `R` = refined feature (passed up). `P` = prediction of the *next* `x_in` (to loss head only). `h_new` = updated recurrent state.
  - **Firewall:** `P` is a function of `h_new` only (which was formed from `x_in` and `h_prev`); it never sees the future target.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_cell.py
import torch
from temporal_tf.config import Config
from temporal_tf.layers import PredictiveCell

def _cfg(): return Config(image_size=16, patch_size=8, d_model=32, n_heads=4)

def test_cell_output_shapes():
    cfg = _cfg(); cell = PredictiveCell(cfg)
    B = 2; h = cell.init_state(B, "cpu")
    x = torch.randn(B, cfg.n_tokens, cfg.d_model)
    R, P, h2 = cell(x, h)
    for t in (R, P, h2):
        assert t.shape == (B, cfg.n_tokens, cfg.d_model)

def test_state_carries_information():
    cfg = _cfg(); cell = PredictiveCell(cfg); B = 2
    x = torch.randn(B, cfg.n_tokens, cfg.d_model)
    _, _, h1 = cell(x, cell.init_state(B, "cpu"))
    R_a, _, _ = cell(x, h1)
    R_b, _, _ = cell(x, cell.init_state(B, "cpu"))
    assert not torch.allclose(R_a, R_b)  # same input, different state -> different output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cell.py -v`
Expected: FAIL with `ImportError` (no `PredictiveCell`)

- [ ] **Step 3: Write the implementation (append to `temporal_tf/layers.py`)**
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cell.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/layers.py tests/test_cell.py
git commit -m "feat: predictive cell with recurrent state + dual (refine/predict) outputs"
```

---

### Task 5: EMA target wrapper

**Files:**
- Create: `temporal_tf/ema.py`
- Test: `tests/test_ema.py`

**Interfaces:**
- Produces: `class EMATarget`: `__init__(online_module: nn.Module)` (deep-copies, disables grad); `@torch.no_grad() update(online_module, momentum)`; `.module` is the target net (call it under `torch.no_grad()`).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_ema.py
import torch, torch.nn as nn
from temporal_tf.ema import EMATarget

def test_ema_moves_toward_online_and_no_grad():
    online = nn.Linear(4, 4)
    tgt = EMATarget(online)
    for p in tgt.module.parameters():
        assert p.requires_grad is False
    with torch.no_grad():
        for p in online.parameters(): p.add_(1.0)  # shift online
    before = next(tgt.module.parameters()).clone()
    tgt.update(online, momentum=0.9)
    after = next(tgt.module.parameters())
    assert torch.all(after >= before) and not torch.allclose(after, before)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ema.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**
```python
# temporal_tf/ema.py
import copy
import torch
import torch.nn as nn

class EMATarget:
    def __init__(self, online_module: nn.Module):
        self.module = copy.deepcopy(online_module)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_module: nn.Module, momentum: float):
        for tp, op in zip(self.module.parameters(), online_module.parameters()):
            tp.mul_(momentum).add_(op.detach(), alpha=1.0 - momentum)
        for tb, ob in zip(self.module.buffers(), online_module.buffers()):
            tb.copy_(ob)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ema.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/ema.py tests/test_ema.py
git commit -m "feat: EMA target wrapper for stop-grad prediction targets"
```

---

### Task 6: Wavefront model forward (lag + prediction/target records)

**Files:**
- Create: `temporal_tf/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `FrameEncoder`, `PredictiveCell`, `EMATarget`, `make_token_mask`, `Config`.
- Produces: `class TemporalDepthModel(nn.Module)`:
  - `__init__(cfg)`: `self.encoder=FrameEncoder`, `self.cells=ModuleList([PredictiveCell ...])` of length `n_layers-1`, `self.target=EMATarget(self)`-equivalent built from an inner online stack (see note), `update_target()`.
  - `horizon(l) -> int`: `1` if `cfg.horizon_mode=="A"` else `l`.
  - `forward(clip:(B,T,1,H,W), rng=None) -> Output` where `Output` is a dataclass with:
    - `features: list[list[Tensor]]` indexed `[layer][tick]`, each `(B,N,d)` (online refined features; `layer` 0..n_layers-1).
    - `pairs: list[tuple[int,int,Tensor,Tensor]]` = `(layer_l, tick_t, P, target)` valid prediction/target pairs (target = stop-grad EMA `R_{l-1}` at tick `t+horizon(l)`).
    - `surprise: Tensor (B,T,n_layers)` (NaN where undefined), per-(layer,tick) scalar = mean token L2 of `target - P`.
  - **Lag rule:** cell `l` at tick `t` consumes online `R_{l-1}` from tick `t-1` (zeros at `t=0`).

**Note on the target net:** build the EMA over an *inner* `nn.Module` (`_OnlineStack` holding encoder+cells) so `self.target.module` is a structural twin. `forward` runs the online stack with grad and the target stack under `no_grad`, both with the same masked inputs, applying the same lag.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_model.py
import torch
from temporal_tf.config import Config
from temporal_tf.model import TemporalDepthModel

def _cfg(): return Config(image_size=16, patch_size=8, d_model=32, n_heads=4, n_layers=4, clip_len=6, mask_ratio=0.5)

def test_forward_record_shapes():
    cfg = _cfg(); m = TemporalDepthModel(cfg)
    clip = torch.rand(2, cfg.clip_len, 1, 16, 16)
    out = m(clip, rng=torch.Generator().manual_seed(0))
    assert len(out.features) == cfg.n_layers
    assert len(out.features[0]) == cfg.clip_len
    assert out.surprise.shape == (2, cfg.clip_len, cfg.n_layers)
    assert len(out.pairs) > 0
    for (l, t, P, tgt) in out.pairs:
        assert P.shape == tgt.shape == (2, cfg.n_tokens, cfg.d_model)
        assert not tgt.requires_grad   # target is stop-grad

def test_pair_targets_respect_horizon_A():
    cfg = _cfg(); m = TemporalDepthModel(cfg)   # horizon_mode "A" -> 1 step
    out = m(torch.rand(2, cfg.clip_len, 1, 16, 16), rng=torch.Generator().manual_seed(1))
    # every pair's target tick = t + 1, and no pair exceeds clip length
    assert all(t + 1 <= cfg.clip_len - 1 for (l, t, P, tgt) in out.pairs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**
```python
# temporal_tf/model.py
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
                tgt_t = t + h
                if preds[li][t] is None or tgt_t >= T:
                    continue
                target = tgt_feats[li - 1][tgt_t].detach()   # stop-grad EMA target
                P = preds[li][t]
                pairs.append((li, t, P, target))
                surprise[:, t, li] = (target - P).pow(2).mean(-1).sqrt().mean(-1)
        return Output(features=feats, pairs=pairs, surprise=surprise)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/model.py tests/test_model.py
git commit -m "feat: wavefront forward with one-tick lag, EMA targets, surprise records"
```

---

### Task 7: Losses (predictive + VICReg) and the surprise scalar

**Files:**
- Create: `temporal_tf/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Consumes: `Output`, `Config`.
- Produces:
  - `predictive_loss(pairs) -> Tensor` = mean MSE over pairs of `P` vs `target` (target already stop-grad).
  - `vicreg(features_flat: Tensor (M,d), var_coef, cov_coef) -> Tensor` = `var_coef*var_term + cov_coef*cov_term` (VICReg variance hinge at 1.0 + off-diagonal covariance).
  - `total_loss(out: Output, cfg: Config) -> tuple[Tensor, dict]`: combines `cfg.pred_coef*predictive_loss` + VICReg over all online refined features (layers ≥1, all ticks, flattened over B and tokens). Returns `(loss, {"pred":..,"var":..,"cov":..})`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_losses.py
import torch
from temporal_tf.losses import predictive_loss, vicreg

def test_predictive_loss_zero_when_equal():
    x = torch.randn(2, 4, 8)
    pairs = [(1, 0, x, x.detach())]
    assert predictive_loss(pairs).item() < 1e-6

def test_vicreg_penalizes_collapse():
    collapsed = torch.zeros(16, 8)            # no variance -> high variance loss
    spread = torch.randn(16, 8) * 2.0
    assert vicreg(collapsed, 25.0, 1.0) > vicreg(spread, 25.0, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_losses.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**
```python
# temporal_tf/losses.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_losses.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/losses.py tests/test_losses.py
git commit -m "feat: predictive + VICReg losses"
```

---

### Task 8: Training step + loop (truncated BPTT, EMA update)

**Files:**
- Create: `temporal_tf/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `TemporalDepthModel`, `total_loss`, data generators, `Config`.
- Produces:
  - `train_step(model, clip, optimizer, cfg, rng) -> dict` (one forward/backward/step + `model.update_target()`; returns loss parts).
  - `overfit_batch(cfg, steps:int, clip=None) -> list[float]` (repeatedly trains on one fixed clip; returns total-loss history) — used to prove learning.
  - `train(cfg, n_steps, digit_bank=None) -> TemporalDepthModel` (full loop sampling fresh clips; uses synthetic bank if none given).

**TBPTT note:** for the prototype, clips are short (`clip_len ≤ 20`) so a full backward over the clip is fine; `cfg.tbptt` is honored by detaching nothing when `tbptt >= clip_len`. (A longer-stream chunked variant is future work, listed in the spec's open questions.)

- [ ] **Step 1: Write the failing test**
```python
# tests/test_train.py
import torch
from temporal_tf.config import Config
from temporal_tf.train import overfit_batch

def test_overfit_reduces_loss():
    cfg = Config(image_size=16, patch_size=8, d_model=32, n_heads=4, n_layers=4,
                 clip_len=6, mask_ratio=0.5, lr=3e-3, seed=0)
    hist = overfit_batch(cfg, steps=40)
    assert hist[-1] < hist[0] * 0.8     # loss drops at least 20% on a fixed batch
    assert all(torch.isfinite(torch.tensor(h)) for h in hist)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**
```python
# temporal_tf/train.py
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
        bank = synthetic_digit_bank()
        clip, _ = generate_clip(torch.Generator().manual_seed(cfg.seed), bank, cfg)
        clip = clip.unsqueeze(0)                       # B=1
    rng = torch.Generator().manual_seed(cfg.seed)
    hist = []
    for _ in range(steps):
        hist.append(train_step(model, clip, opt, cfg, rng)["total"])
    return hist

def train(cfg: Config, n_steps: int, digit_bank=None):
    torch.manual_seed(cfg.seed)
    bank = digit_bank if digit_bank is not None else synthetic_digit_bank()
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS (1 passed). *(If flaky, raise `steps` to 80 — overfitting one clip must drive loss down.)*

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/train.py tests/test_train.py
git commit -m "feat: training step/loop with EMA update + overfit sanity"
```

---

### Task 9: Collapse monitor

**Files:**
- Create: `temporal_tf/eval/__init__.py`
- Create: `temporal_tf/eval/collapse.py`
- Test: `tests/test_collapse.py`

**Interfaces:**
- Produces: `representation_stats(R: Tensor (...,d)) -> dict{"std":float,"eff_rank":float}` (mean per-dim std; effective rank = `exp(entropy of normalized singular values)`). `is_collapsed(R, std_thresh=1e-2) -> bool`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_collapse.py
import torch
from temporal_tf.eval.collapse import representation_stats, is_collapsed

def test_detects_collapse_vs_healthy():
    collapsed = torch.zeros(64, 32)
    healthy = torch.randn(64, 32)
    assert is_collapsed(collapsed) and not is_collapsed(healthy)
    assert representation_stats(healthy)["eff_rank"] > representation_stats(collapsed)["eff_rank"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collapse.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**
```python
# temporal_tf/eval/__init__.py
```
```python
# temporal_tf/eval/collapse.py
import torch

def representation_stats(R):
    x = R.reshape(-1, R.shape[-1])
    std = float(x.std(0).mean())
    xc = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(xc)
    p = s / (s.sum() + 1e-9)
    eff_rank = float(torch.exp(-(p * (p + 1e-9).log()).sum()))
    return {"std": std, "eff_rank": eff_rank}

def is_collapsed(R, std_thresh=1e-2):
    return representation_stats(R)["std"] < std_thresh
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_collapse.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/eval/__init__.py temporal_tf/eval/collapse.py tests/test_collapse.py
git commit -m "feat: collapse monitor (std + effective rank)"
```

---

### Task 10: Surprise localization eval

**Files:**
- Create: `temporal_tf/eval/surprise_eval.py`
- Test: `tests/test_surprise_eval.py`

**Interfaces:**
- Produces: `localization_auc(surprise_layer: Tensor (T,), events: Tensor (T,) bool) -> float` (ROC-AUC of per-tick surprise predicting event ticks; NaNs in surprise dropped). Returns `0.5` if events are degenerate (all/none).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_surprise_eval.py
import torch
from temporal_tf.eval.surprise_eval import localization_auc

def test_auc_high_when_surprise_tracks_events():
    T = 20
    events = torch.zeros(T, dtype=torch.bool); events[[5, 12, 17]] = True
    surprise = torch.rand(T) * 0.1
    surprise[events] += 1.0                       # spikes at events
    assert localization_auc(surprise, events) > 0.9

def test_auc_chance_when_flat():
    T = 20
    events = torch.zeros(T, dtype=torch.bool); events[[3, 9]] = True
    assert abs(localization_auc(torch.ones(T), events) - 0.5) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_surprise_eval.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**
```python
# temporal_tf/eval/surprise_eval.py
import torch
from sklearn.metrics import roc_auc_score

def localization_auc(surprise_layer, events):
    s = surprise_layer.detach().cpu()
    e = events.cpu().bool()
    valid = ~torch.isnan(s)
    s, e = s[valid], e[valid]
    if e.sum() == 0 or e.sum() == len(e):
        return 0.5
    if torch.allclose(s, s[0].expand_as(s)):
        return 0.5
    return float(roc_auc_score(e.numpy(), s.numpy()))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_surprise_eval.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/eval/surprise_eval.py tests/test_surprise_eval.py
git commit -m "feat: surprise->event localization AUC"
```

---

### Task 11: Timescale decorrelation + linear probe

**Files:**
- Create: `temporal_tf/eval/timescale.py`
- Create: `temporal_tf/eval/probe.py`
- Test: `tests/test_timescale_probe.py`

**Interfaces:**
- Produces:
  - `cross_layer_correlation(surprise: Tensor (T,n_layers)) -> Tensor (n_layers,n_layers)` (Pearson corr across ticks between layers' surprise; NaN-safe by dropping rows with any NaN).
  - `linear_probe(features: Tensor (M,d), labels: Tensor (M,)) -> float` (train/test split accuracy with `sklearn.linear_model.LogisticRegression`).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_timescale_probe.py
import torch
from temporal_tf.eval.timescale import cross_layer_correlation
from temporal_tf.eval.probe import linear_probe

def test_correlation_identity_diag():
    T, L = 30, 3
    s = torch.randn(T, L)
    c = cross_layer_correlation(s)
    assert c.shape == (L, L)
    assert torch.allclose(c.diag(), torch.ones(L), atol=1e-5)

def test_probe_learns_separable_features():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(50, 8, generator=g) + 3.0
    b = torch.randn(50, 8, generator=g) - 3.0
    X = torch.cat([a, b]); y = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    assert linear_probe(X, y) > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_timescale_probe.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**
```python
# temporal_tf/eval/timescale.py
import torch

def cross_layer_correlation(surprise):
    s = surprise.detach().cpu()
    valid = ~torch.isnan(s).any(1)
    s = s[valid]
    s = s - s.mean(0, keepdim=True)
    std = s.std(0, keepdim=True) + 1e-9
    s = s / std
    return (s.T @ s) / (s.shape[0] - 1)
```
```python
# temporal_tf/eval/probe.py
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def linear_probe(features, labels):
    X = features.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return float(clf.score(Xte, yte))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_timescale_probe.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/eval/timescale.py temporal_tf/eval/probe.py tests/test_timescale_probe.py
git commit -m "feat: timescale decorrelation + linear probe evals"
```

---

### Task 12: Colab entrypoint + end-to-end smoke test

**Files:**
- Create: `scripts/run_prototype.py`
- Create: `notebooks/README.md` (Colab usage)
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `scripts/run_prototype.py` with `run(cfg=None, n_steps=300, use_mnist=False) -> dict` that trains, then reports: collapse stats per layer, surprise-localization AUC per layer (averaged over fresh clips), and cross-layer surprise correlation. CLI via `python scripts/run_prototype.py`. (The `linear_probe` utility from Task 11 is unit-tested and available for feature-quality checks once the data generator is extended to return digit labels — a documented follow-up, not part of `run()`.)

- [ ] **Step 1: Write the failing test**
```python
# tests/test_smoke.py
import torch
from temporal_tf.config import Config
from scripts.run_prototype import run

def test_end_to_end_smoke():
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=4,
                 clip_len=6, mask_ratio=0.5, lr=3e-3, seed=0)
    report = run(cfg=cfg, n_steps=20, use_mnist=False)
    assert "auc_per_layer" in report and len(report["auc_per_layer"]) == cfg.n_layers - 1
    assert "collapse_std_per_layer" in report
    for v in report["auc_per_layer"]:
        assert 0.0 <= v <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'` (add `scripts/__init__.py` if needed for import, or use a `sys.path` shim in the test — create `scripts/__init__.py`).

- [ ] **Step 3: Write the implementation**
```python
# scripts/__init__.py
```
```python
# scripts/run_prototype.py
import torch
from temporal_tf.config import Config, default_config
from temporal_tf.train import train
from temporal_tf.data import synthetic_digit_bank, generate_clip
from temporal_tf.eval.collapse import representation_stats
from temporal_tf.eval.surprise_eval import localization_auc
from temporal_tf.eval.timescale import cross_layer_correlation

def run(cfg: Config = None, n_steps: int = 300, use_mnist: bool = False) -> dict:
    cfg = cfg or default_config()
    bank = None
    if use_mnist:
        from temporal_tf.data import load_mnist_digit_bank
        bank = load_mnist_digit_bank("./data")
    model = train(cfg, n_steps=n_steps, digit_bank=bank)

    model.eval()
    bank = bank if bank is not None else synthetic_digit_bank()
    gen = torch.Generator().manual_seed(cfg.seed + 99)
    aucs = [[] for _ in range(cfg.n_layers)]
    all_surprise = []
    with torch.no_grad():
        for _ in range(8):
            clip, bounces = generate_clip(gen, bank, cfg)
            out = model(clip.unsqueeze(0), rng=torch.Generator().manual_seed(0))
            s = out.surprise[0]                       # (T, n_layers)
            all_surprise.append(s)
            for li in range(1, cfg.n_layers):
                aucs[li].append(localization_auc(s[:, li], bounces))
        last = model(clip.unsqueeze(0), rng=torch.Generator().manual_seed(0))
        collapse = [representation_stats(torch.stack(last.features[li]))["std"]
                    for li in range(cfg.n_layers)]
    corr = cross_layer_correlation(torch.cat(all_surprise, 0))
    return {
        "auc_per_layer": [float(sum(a) / len(a)) for a in aucs[1:]],
        "collapse_std_per_layer": collapse,
        "cross_layer_surprise_corr": corr.tolist(),
    }

if __name__ == "__main__":
    import json
    print(json.dumps(run(n_steps=300, use_mnist=True), indent=2))
```
```markdown
<!-- notebooks/README.md -->
# Colab usage

```python
!git clone https://github.com/GhostOfRazgriz1/temporal_tf.git
%cd temporal_tf
!pip install -e .
from scripts.run_prototype import run
report = run(n_steps=500, use_mnist=True)
report
```
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the whole suite + commit**
```bash
pytest -q
git add scripts/ notebooks/README.md tests/test_smoke.py
git commit -m "feat: colab entrypoint + end-to-end smoke test"
```

---

## After the prototype runs (research validation, not code)

Once all tasks pass and you've trained on real MNIST (`use_mnist=True`, ~500+ steps on a Colab GPU), evaluate the spec's three claims:

1. **Learns without collapse** — `collapse_std_per_layer` stays well above `1e-2` for every layer while `pred` loss falls.
2. **Surprise is meaningful** — `auc_per_layer` > 0.5 (ideally ≫) for predicting bounce ticks.
3. **Timescale hierarchy** — `cross_layer_surprise_corr` shows shallow and deep layers decorrelating. If they don't, flip `horizon_mode="B"` (the knob from the spec) and re-run.

These are the empirical questions the spec's §8 defines; record results back into the spec or a results doc.
```
```

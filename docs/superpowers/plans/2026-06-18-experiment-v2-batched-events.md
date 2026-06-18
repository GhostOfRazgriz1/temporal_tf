# Experiment v2 — Batched training, stochastic events, cleaner surprise eval

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development. Steps use `- [ ]`.

**Goal:** Make the prototype's surprise hypothesis actually testable: train at real batch size, inject genuinely-unpredictable teleport events, and evaluate surprise localization (per event type, mask-averaged, over many clips) — plus an optional AUC-over-training curve.

**Architecture:** Extends the existing `temporal_tf` module. New `Config` fields all default to current behavior (existing 21 tests must stay green). Adds a shared `_simulate` in `data.py` (bounce + teleport labels), a `generate_batch`, batched training in `train.py`, and a v2 `run()`/eval in `run_prototype.py`.

**Tech Stack:** Python 3.13 venv at `./.venv`, PyTorch CPU, pytest. Run tests via `./.venv/Scripts/python.exe -m pytest`.

## Global Constraints
- **Backward compatibility:** `generate_clip(rng, bank, cfg, n_digits=1) -> (clip, bounces)` MUST keep its exact 2-tuple signature/behavior; new `Config` fields default to current behavior (`teleport_prob=0.0`, `batch_size=1`, `n_mask_draws=1`, `n_eval_clips=8`). The existing suite must remain green.
- **Determinism:** all randomness flows through the passed `torch.Generator` (`generator=rng`); no unseeded `torch.rand`.
- **Teleport semantics:** on a teleport tick, all digits jump to fresh random positions, `teleport[t]=True`, and that tick records NO bounce.
- **Device:** training already moves `model.online` and `model.target.module` to device; preserve that.
- Tensor convention `(B, n_tokens, d_model)`; clips `(B, T, 1, H, W)`.

---

### Task 1: Teleport events + batched generator + Config fields

**Files:**
- Modify: `temporal_tf/config.py` (add fields)
- Modify: `temporal_tf/data.py` (refactor to `_simulate`; add `generate_batch`; keep `generate_clip`)
- Test: `tests/test_data_v2.py`

**Interfaces:**
- `Config` gains: `teleport_prob: float = 0.0`, `batch_size: int = 1`, `n_mask_draws: int = 1`, `n_eval_clips: int = 8`.
- `_simulate(rng, digit_bank, cfg, n_digits, teleport_prob) -> (clip (T,1,H,W), bounces (T,) bool, teleports (T,) bool)`.
- `generate_clip(rng, digit_bank, cfg, n_digits=1) -> (clip, bounces)` (unchanged signature; delegates to `_simulate` with `teleport_prob=0.0`).
- `generate_batch(rng, digit_bank, cfg, B, n_digits=1) -> (clips (B,T,1,H,W), events: dict{"bounce":(B,T) bool, "teleport":(B,T) bool})` (uses `cfg.teleport_prob`).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_data_v2.py
import torch
from temporal_tf.config import Config
from temporal_tf.data import synthetic_digit_bank, generate_clip, generate_batch, _simulate

def _cfg(**kw): return Config(image_size=32, patch_size=8, clip_len=12, **kw)

def test_generate_clip_unchanged_signature():
    cfg = _cfg(); bank = synthetic_digit_bank()
    clip, bounces = generate_clip(torch.Generator().manual_seed(0), bank, cfg)
    assert clip.shape == (cfg.clip_len, 1, 32, 32) and bounces.dtype == torch.bool

def test_teleport_off_by_default():
    cfg = _cfg(teleport_prob=0.0); bank = synthetic_digit_bank()
    _, _, tel = _simulate(torch.Generator().manual_seed(1), bank, cfg, 1, 0.0)
    assert tel.sum() == 0

def test_teleport_always_when_prob_one():
    cfg = _cfg(); bank = synthetic_digit_bank()
    _, bounces, tel = _simulate(torch.Generator().manual_seed(2), bank, cfg, 1, 1.0)
    assert bool(tel.all()) and bounces.sum() == 0   # all teleport, no bounces

def test_generate_batch_shapes_and_events():
    cfg = _cfg(teleport_prob=0.3); bank = synthetic_digit_bank()
    clips, ev = generate_batch(torch.Generator().manual_seed(3), bank, cfg, B=4)
    assert clips.shape == (4, cfg.clip_len, 1, 32, 32)
    assert ev["bounce"].shape == ev["teleport"].shape == (4, cfg.clip_len)
    assert ev["bounce"].dtype == torch.bool
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_data_v2.py -v`
Expected: FAIL with `ImportError` (no `generate_batch`/`_simulate`)

- [ ] **Step 3: Implement**

Add to `temporal_tf/config.py` (inside `Config`, after existing fields, before the `n_tokens` property):
```python
    teleport_prob: float = 0.0
    batch_size: int = 1
    n_mask_draws: int = 1
    n_eval_clips: int = 8
```

Rewrite `temporal_tf/data.py`'s `generate_clip` into a shared `_simulate` + thin wrappers (keep `synthetic_digit_bank` and `load_mnist_digit_bank` as they are):
```python
def _simulate(rng, digit_bank, cfg, n_digits, teleport_prob):
    H = W = cfg.image_size
    ds = digit_bank.shape[-1]
    T = cfg.clip_len
    clip = torch.zeros(T, 1, H, W)
    bounces = torch.zeros(T, dtype=torch.bool)
    teleports = torch.zeros(T, dtype=torch.bool)
    idx = torch.randint(0, digit_bank.shape[0], (n_digits,), generator=rng)
    limits = torch.tensor([H - ds, W - ds]).float()
    pos = torch.rand(n_digits, 2, generator=rng) * limits
    vel = (torch.rand(n_digits, 2, generator=rng) * 2 - 1) * 4.0
    for t in range(T):
        do_tp = teleport_prob > 0.0 and float(torch.rand(1, generator=rng)) < teleport_prob
        if do_tp:
            pos = torch.rand(n_digits, 2, generator=rng) * limits
            teleports[t] = True
        else:
            for d in range(n_digits):
                for ax, limit in enumerate((H - ds, W - ds)):
                    pos[d, ax] += vel[d, ax]
                    if pos[d, ax] < 0 or pos[d, ax] > limit:
                        vel[d, ax] = -vel[d, ax]
                        pos[d, ax] = pos[d, ax].clamp(0, limit)
                        bounces[t] = True
        for d in range(n_digits):
            y, x = int(pos[d, 0]), int(pos[d, 1])
            patch = digit_bank[idx[d]]
            clip[t, 0, y:y + ds, x:x + ds] = torch.maximum(clip[t, 0, y:y + ds, x:x + ds], patch)
    return clip.clamp(0, 1), bounces, teleports

def generate_clip(rng, digit_bank, cfg, n_digits=1):
    clip, bounces, _ = _simulate(rng, digit_bank, cfg, n_digits, 0.0)
    return clip, bounces

def generate_batch(rng, digit_bank, cfg, B, n_digits=1):
    clips, bounces, teleports = [], [], []
    for _ in range(B):
        c, b, tel = _simulate(rng, digit_bank, cfg, n_digits, cfg.teleport_prob)
        clips.append(c); bounces.append(b); teleports.append(tel)
    events = {"bounce": torch.stack(bounces), "teleport": torch.stack(teleports)}
    return torch.stack(clips), events
```

- [ ] **Step 4: Run tests (new + full suite for regressions)**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_data_v2.py tests/test_data.py -v`
Expected: PASS (4 new + 3 existing). Then `./.venv/Scripts/python.exe -m pytest -q` → all green.

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/config.py temporal_tf/data.py tests/test_data_v2.py
git commit -m "feat: teleport events + batched data generator + experiment config fields"
```

---

### Task 2: Batched training + AUC-over-training hook

**Files:**
- Modify: `temporal_tf/train.py` (`train` uses `generate_batch` + `cfg.batch_size`; add `on_eval`/`eval_every`)
- Test: `tests/test_train_v2.py`

**Interfaces:**
- `train(cfg, n_steps, digit_bank=None, device=None, on_eval=None, eval_every=0) -> model` (unchanged return = model). Each step samples `generate_batch(gen, bank, cfg, cfg.batch_size)` and trains on the batched clips (moved to device). If `on_eval` is given and `eval_every>0`, calls `on_eval(model, step)` at steps where `step % eval_every == 0`.
- `train_step`, `overfit_batch` unchanged.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_train_v2.py
import torch
from temporal_tf.config import Config
from temporal_tf.train import train

def _cfg(): return Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=3,
                          clip_len=4, mask_ratio=0.5, batch_size=3, seed=0)

def test_train_runs_batched_and_calls_on_eval():
    cfg = _cfg()
    seen = []
    model = train(cfg, n_steps=4, on_eval=lambda m, s: seen.append(s), eval_every=2)
    assert seen == [0, 2]                       # called at steps 0 and 2
    assert next(model.online.parameters()).requires_grad
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_train_v2.py -v`
Expected: FAIL (`train()` has no `on_eval`/`eval_every` params)

- [ ] **Step 3: Implement** — replace `train` in `temporal_tf/train.py` with:
```python
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
```
Add `generate_batch` to the existing data import line in `train.py`:
`from .data import synthetic_digit_bank, generate_clip, generate_batch`

- [ ] **Step 4: Run tests (new + full suite)**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_train_v2.py tests/test_train.py -v` then `./.venv/Scripts/python.exe -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/train.py tests/test_train_v2.py
git commit -m "feat: batched training loop + on_eval hook"
```

---

### Task 3: v2 surprise eval (mask-averaged, per-event-type AUC) + run() report + training curve

**Files:**
- Modify: `temporal_tf/run_prototype.py` (add `eval_surprise` helper; rewrite `run`)
- Modify: `tests/test_smoke.py` (assert v2 keys)

**Interfaces:**
- `eval_surprise(model, cfg, bank, n_clips, n_mask_draws) -> dict` with keys:
  - `auc_bounce_per_layer: list[float]` (length `n_layers-1`),
  - `auc_teleport_per_layer: list[float]` (length `n_layers-1`),
  - `cross_layer_surprise_corr: list[list[float]]`.
  Surprise per clip is averaged over `n_mask_draws` forward passes (different mask seeds), then per-layer AUC vs each event type is averaged over clips.
- `run(cfg=None, n_steps=300, use_mnist=False, track_every=0) -> dict` returns the `eval_surprise` keys plus `collapse_std_per_layer` and, if `track_every>0`, `training_curve: list[dict]` (each `{"step", "auc_bounce_per_layer", "auc_teleport_per_layer"}`).

- [ ] **Step 1: Write the failing test** (extend smoke)
```python
# tests/test_smoke.py  (add this test; keep the existing one)
def test_end_to_end_smoke_v2():
    from temporal_tf.config import Config
    from temporal_tf.run_prototype import run
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=4,
                 clip_len=6, mask_ratio=0.5, lr=3e-3, seed=0,
                 batch_size=2, teleport_prob=0.3, n_mask_draws=2, n_eval_clips=3)
    report = run(cfg=cfg, n_steps=6, use_mnist=False, track_every=3)
    for k in ("auc_bounce_per_layer", "auc_teleport_per_layer", "collapse_std_per_layer",
              "cross_layer_surprise_corr", "training_curve"):
        assert k in report
    assert len(report["auc_bounce_per_layer"]) == cfg.n_layers - 1
    assert len(report["auc_teleport_per_layer"]) == cfg.n_layers - 1
    assert len(report["training_curve"]) == 2          # steps 0 and 3
    for v in report["auc_teleport_per_layer"]:
        assert 0.0 <= v <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_smoke.py::test_end_to_end_smoke_v2 -v`
Expected: FAIL (`run()` lacks `track_every` / new keys)

- [ ] **Step 3: Implement** — rewrite `temporal_tf/run_prototype.py`:
```python
import torch
from temporal_tf.config import Config, default_config
from temporal_tf.train import train
from temporal_tf.data import synthetic_digit_bank, generate_batch
from temporal_tf.eval.collapse import representation_stats
from temporal_tf.eval.surprise_eval import localization_auc
from temporal_tf.eval.timescale import cross_layer_correlation

def eval_surprise(model, cfg, bank, n_clips, n_mask_draws):
    model.eval()
    device = next(model.online.parameters()).device
    gen = torch.Generator().manual_seed(cfg.seed + 99)
    n_pred = cfg.n_layers - 1
    auc_b = [[] for _ in range(n_pred)]
    auc_t = [[] for _ in range(n_pred)]
    all_surprise = []
    with torch.no_grad():
        for _ in range(n_clips):
            clips, ev = generate_batch(gen, bank, cfg, B=1)
            clip = clips.to(device)
            draws = []
            for d in range(max(1, n_mask_draws)):
                out = model(clip, rng=torch.Generator().manual_seed(d))
                draws.append(out.surprise[0])           # (T, n_layers)
            s = torch.stack(draws).mean(0)              # (T, n_layers), mask-averaged
            all_surprise.append(s)
            for li in range(1, cfg.n_layers):
                auc_b[li - 1].append(localization_auc(s[:, li], ev["bounce"][0]))
                auc_t[li - 1].append(localization_auc(s[:, li], ev["teleport"][0]))
    corr = cross_layer_correlation(torch.cat(all_surprise, 0)[:, 1:])
    return {
        "auc_bounce_per_layer": [float(sum(a) / len(a)) for a in auc_b],
        "auc_teleport_per_layer": [float(sum(a) / len(a)) for a in auc_t],
        "cross_layer_surprise_corr": corr.tolist(),
    }

def run(cfg: Config = None, n_steps: int = 300, use_mnist: bool = False, track_every: int = 0) -> dict:
    cfg = cfg or default_config()
    bank = None
    if use_mnist:
        from temporal_tf.data import load_mnist_digit_bank
        bank = load_mnist_digit_bank("./data")
    eval_bank = bank if bank is not None else synthetic_digit_bank(size=cfg.image_size // 2)

    curve = []
    def on_eval(model, step):
        m = eval_surprise(model, cfg, eval_bank, n_clips=max(2, cfg.n_eval_clips // 4), n_mask_draws=1)
        curve.append({"step": step,
                      "auc_bounce_per_layer": m["auc_bounce_per_layer"],
                      "auc_teleport_per_layer": m["auc_teleport_per_layer"]})

    model = train(cfg, n_steps=n_steps, digit_bank=bank,
                  on_eval=(on_eval if track_every > 0 else None), eval_every=track_every)

    full = eval_surprise(model, cfg, eval_bank, n_clips=cfg.n_eval_clips, n_mask_draws=cfg.n_mask_draws)
    device = next(model.online.parameters()).device
    with torch.no_grad():
        clips, _ = generate_batch(torch.Generator().manual_seed(cfg.seed + 7), eval_bank, cfg, B=1)
        last = model(clips.to(device), rng=torch.Generator().manual_seed(0))
        collapse = [representation_stats(torch.stack(last.features[li]))["std"] for li in range(cfg.n_layers)]
    out = {**full, "collapse_std_per_layer": collapse}
    if track_every > 0:
        out["training_curve"] = curve
    return out

if __name__ == "__main__":
    import json
    print(json.dumps(run(n_steps=500, use_mnist=True, track_every=50), indent=2))
```
Also update `notebooks/README.md` example to `run(n_steps=500, use_mnist=True, track_every=50)`.

- [ ] **Step 4: Run tests (new + full suite)**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_smoke.py -v` then `./.venv/Scripts/python.exe -m pytest -q`
Expected: all PASS (including the original smoke test, which calls `run()` with default `track_every=0` → no `training_curve` key; that test only checks the keys it asserts, so keep the original assertions valid — the original `test_end_to_end_smoke` asserts `auc_per_layer`; UPDATE it to assert `auc_bounce_per_layer` instead since `run()`'s shape changed).

- [ ] **Step 5: Reconcile the original smoke test + commit**
Update the ORIGINAL `test_end_to_end_smoke` to use the new key name (`auc_bounce_per_layer` in place of `auc_per_layer`), keeping its other assertions. Then:
```bash
./.venv/Scripts/python.exe -m pytest -q
git add temporal_tf/run_prototype.py tests/test_smoke.py notebooks/README.md
git commit -m "feat: v2 surprise eval (mask-averaged, per-event AUC) + training curve"
```

---

## After this iteration runs
Re-run on real MNIST in Colab with e.g. `Config(teleport_prob=0.1, batch_size=16, n_mask_draws=4, n_eval_clips=50)` and `run(n_steps=10000, use_mnist=True, track_every=200)`. Read: does `auc_teleport_per_layer` rise meaningfully above 0.5 (the clean test of the hypothesis)? Does `auc_bounce_per_layer` rise-then-plateau? Does `training_curve` show the signal emerging? Does cross-layer corr show timescale structure?

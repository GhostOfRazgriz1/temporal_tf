# Experiment v4 — Training stabilization + probe-over-training + multi-seed

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development. Steps use `- [ ]`.

**Goal:** Convert the encouraging-but-messy v3 result into a trustworthy verdict. (1) Stabilize training (gradient clipping + LR warmup/cosine) so `pred` stays low instead of spiking at ~step 2400. (2) Track the error/feature probe AUC **over training** (not just loss) so we see where the teleport signal lives and pick the best checkpoint. (3) Add `run_seeds` to report mean ± std across seeds.

**Why:** v3 showed the model *does* learn (`pred`→0.03) and teleport is decodable from prediction error (error-probe 0.63 > feature 0.566), but training destabilized and the probe was scored at a suboptimal final step. We need stable training + signal-over-time + replication.

**Tech Stack:** Python 3.13 venv `./.venv`, PyTorch CPU, scikit-learn, pytest. Tests via `./.venv/Scripts/python.exe -m pytest`.

## Global Constraints
- **Backward compatibility:** new `Config` fields default to current behavior (`grad_clip=0.0` → no clipping; `warmup_steps=0` + `lr_schedule="none"` → constant LR). Existing tests stay green.
- `train()` still returns `model`; `on_eval(model, step, parts)` stays 3-arg.
- Determinism via passed `torch.Generator`; probes use fixed `random_state`.
- `run_seeds` must not mutate the passed `cfg` (use `dataclasses.replace`).

---

### Task 1: Training stabilization (grad clip + LR schedule)

**Files:**
- Modify: `temporal_tf/config.py` (add fields)
- Modify: `temporal_tf/train.py` (grad clip in `train_step`; LR schedule in `train`; `_lr_lambda` helper)
- Test: `tests/test_stabilization.py`

**Interfaces:**
- `Config` gains: `grad_clip: float = 0.0`, `warmup_steps: int = 0`, `lr_schedule: str = "none"` (`"none"|"cosine"`).
- `_lr_lambda(step: int, cfg: Config, n_steps: int) -> float` — warmup ramp `(step+1)/warmup_steps` while `step < warmup_steps`; then `1.0` if `lr_schedule=="none"`, else cosine `0.5*(1+cos(pi*progress))` with `progress=(step-warmup_steps)/max(1,n_steps-warmup_steps)` clamped to [0,1].
- `train_step` clips gradients to `cfg.grad_clip` (max-norm over `model.online.parameters()`) when `grad_clip > 0`, after `backward()` and before `optimizer.step()`.
- `train` builds a `LambdaLR(opt, lambda s: _lr_lambda(s, cfg, n_steps))` and calls `sched.step()` once per training step.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_stabilization.py
import math
import torch
from temporal_tf.config import Config
from temporal_tf.train import _lr_lambda, train

def test_lr_lambda_warmup_then_cosine():
    cfg = Config(warmup_steps=10, lr_schedule="cosine", lr=1e-3)
    assert abs(_lr_lambda(0, cfg, 100) - 0.1) < 1e-6          # first warmup step
    assert abs(_lr_lambda(9, cfg, 100) - 1.0) < 1e-6          # end of warmup
    mid = _lr_lambda(55, cfg, 100)                            # halfway through cosine
    assert 0.3 < mid < 0.7
    assert _lr_lambda(100, cfg, 100) < 1e-3                   # decays to ~0 at the end

def test_lr_lambda_none_is_constant_after_warmup():
    cfg = Config(warmup_steps=0, lr_schedule="none")
    assert _lr_lambda(0, cfg, 100) == 1.0 and _lr_lambda(50, cfg, 100) == 1.0

def test_train_runs_stabilized():
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=3,
                 clip_len=4, mask_ratio=0.5, batch_size=2, grad_clip=1.0,
                 warmup_steps=2, lr_schedule="cosine", seed=0)
    model = train(cfg, n_steps=5)
    for p in model.online.parameters():
        assert torch.isfinite(p).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_stabilization.py -v`
Expected: FAIL (`_lr_lambda` missing; `Config` lacks fields).

- [ ] **Step 3: Implement**

Add to `temporal_tf/config.py` (with the other fields, before `n_tokens`):
```python
    grad_clip: float = 0.0
    warmup_steps: int = 0
    lr_schedule: str = "none"   # "none" | "cosine"
```

In `temporal_tf/train.py`, add `import math` at the top, then add the helper and update `train_step` + `train`:
```python
def _lr_lambda(step, cfg, n_steps):
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    if cfg.lr_schedule == "cosine":
        denom = max(1, n_steps - cfg.warmup_steps)
        progress = min(1.0, max(0.0, (step - cfg.warmup_steps) / denom))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0
```
In `train_step`, insert clipping between `loss.backward()` and `optimizer.step()`:
```python
    loss.backward()
    if cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.online.parameters(), cfg.grad_clip)
    optimizer.step()
```
In `train`, after building `opt`, add the scheduler and step it each iteration:
```python
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_lambda(s, cfg, n_steps))
    ...
    for step in range(n_steps):
        ...
        parts = train_step(model, clips.to(device), opt, cfg, rng)
        sched.step()
        if on_eval is not None and eval_every > 0 and step % eval_every == 0:
            on_eval(model, step, parts)
        ...
```

- [ ] **Step 4: Run tests (new + full suite)**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_stabilization.py -v` then `./.venv/Scripts/python.exe -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/config.py temporal_tf/train.py tests/test_stabilization.py
git commit -m "feat: training stabilization (grad clip + LR warmup/cosine schedule)"
```

---

### Task 2: Probe-over-training + best-checkpoint + run_seeds

**Files:**
- Modify: `temporal_tf/run_prototype.py` (probe AUC in `on_eval`; `best_error_probe_teleport` in report; add `run_seeds`)
- Modify: `tests/test_smoke.py` (assert v4 curve/keys)
- Test: `tests/test_run_seeds.py`

**Interfaces:**
- `run(...)` `training` behavior: when `track_every>0`, each logged `loss_curve` entry ALSO carries `"error_probe_auc": {"bounce","teleport"}` and `"feature_probe_auc": {"bounce","teleport"}`, computed on a small eval set (`max(2, n_eval_clips//5)` clips). The report additionally returns `best_error_probe_teleport: {"step": int, "auc": float}` = the curve entry maximizing `error_probe_auc["teleport"]` (omitted if `track_every==0`).
- `run_seeds(cfg, n_steps, seeds: list[int], use_mnist=False, track_every=0) -> dict` — runs `run` for each seed (via `dataclasses.replace(cfg, seed=s)`), returns `{"final_error_probe_teleport": {"mean","std","values"}, "final_error_probe_bounce": {...}}` and, when `track_every>0`, `"best_error_probe_teleport": {"mean","std","values"}`.

- [ ] **Step 1: Write the failing tests**
```python
# tests/test_run_seeds.py
from temporal_tf.config import Config
from temporal_tf.run_prototype import run_seeds

def test_run_seeds_aggregates():
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=3,
                 clip_len=6, mask_ratio=0.5, batch_size=2, teleport_prob=0.3,
                 n_mask_draws=1, n_eval_clips=4, surprise_topk=2, lr=3e-3)
    agg = run_seeds(cfg, n_steps=4, seeds=[0, 1], use_mnist=False)
    m = agg["final_error_probe_teleport"]
    assert set(m) == {"mean", "std", "values"} and len(m["values"]) == 2
    assert 0.0 <= m["mean"] <= 1.0
```
```python
# add to tests/test_smoke.py (keep the v3 test or update it to v4 keys)
def test_smoke_tracks_probe_over_training():
    from temporal_tf.config import Config
    from temporal_tf.run_prototype import run
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=4,
                 clip_len=6, mask_ratio=0.5, lr=3e-3, seed=0, batch_size=2,
                 teleport_prob=0.3, n_mask_draws=1, n_eval_clips=5, surprise_topk=2)
    report = run(cfg=cfg, n_steps=6, use_mnist=False, track_every=3)
    assert "best_error_probe_teleport" in report
    assert set(report["best_error_probe_teleport"]) == {"step", "auc"}
    entry = report["loss_curve"][0]
    assert "error_probe_auc" in entry and "teleport" in entry["error_probe_auc"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_run_seeds.py tests/test_smoke.py::test_smoke_tracks_probe_over_training -v`
Expected: FAIL (`run_seeds` missing; curve lacks probe keys).

- [ ] **Step 3: Implement** — update `temporal_tf/run_prototype.py`. Add `import statistics` and `from dataclasses import replace` at the top. Replace the `on_eval`/report section of `run` and append `run_seeds`:
```python
def _probe_set(model, cfg, bank, n_clips):
    red, err_feats, feat_feats, labels, _ = _collect(model, cfg, bank, n_clips)
    err_X, feat_X = torch.cat(err_feats), torch.cat(feat_feats)
    err = {ev: event_probe(err_X, torch.cat(labels[ev])) for ev in ("bounce", "teleport")}
    feat = {ev: event_probe(feat_X, torch.cat(labels[ev])) for ev in ("bounce", "teleport")}
    return err, feat

def run(cfg: Config = None, n_steps: int = 300, use_mnist: bool = False, track_every: int = 0) -> dict:
    cfg = cfg or default_config()
    bank = None
    if use_mnist:
        from temporal_tf.data import load_mnist_digit_bank
        bank = load_mnist_digit_bank("./data")
    eval_bank = bank if bank is not None else synthetic_digit_bank(size=cfg.image_size // 2)

    loss_curve = []
    probe_clips = max(2, cfg.n_eval_clips // 5)
    def on_eval(model, step, parts):
        err, feat = _probe_set(model, cfg, eval_bank, probe_clips)
        loss_curve.append({"step": step, "pred": parts["pred"], "var": parts["var"],
                           "cov": parts["cov"], "total": parts["total"],
                           "error_probe_auc": err, "feature_probe_auc": feat})

    model = train(cfg, n_steps=n_steps, digit_bank=bank,
                  on_eval=(on_eval if track_every > 0 else None), eval_every=track_every)

    red, err_feats, feat_feats, labels, last = _collect(model, cfg, eval_bank, cfg.n_eval_clips)
    raw = {}
    for ev in ("bounce", "teleport"):
        y = torch.cat(labels[ev])
        raw[ev] = {r: [pooled_auc(torch.cat(red[r])[:, li], y) for li in range(1, cfg.n_layers)]
                   for r in ("mean", "max", "topk")}
    err_X, feat_X = torch.cat(err_feats), torch.cat(feat_feats)
    error_probe = {ev: event_probe(err_X, torch.cat(labels[ev])) for ev in ("bounce", "teleport")}
    feature_probe = {ev: event_probe(feat_X, torch.cat(labels[ev])) for ev in ("bounce", "teleport")}
    collapse = [representation_stats(torch.stack(last.features[li]))["std"] for li in range(cfg.n_layers)]

    out = {
        "loss_curve": loss_curve,
        "raw_pooled_auc": raw,
        "error_probe_auc": error_probe,
        "feature_probe_auc": feature_probe,
        "collapse_std_per_layer": collapse,
    }
    if track_every > 0 and loss_curve:
        best = max(loss_curve, key=lambda e: e["error_probe_auc"]["teleport"])
        out["best_error_probe_teleport"] = {"step": best["step"], "auc": best["error_probe_auc"]["teleport"]}
    return out

def _ms(xs):
    return {"mean": statistics.fmean(xs),
            "std": (statistics.pstdev(xs) if len(xs) > 1 else 0.0),
            "values": list(xs)}

def run_seeds(cfg: Config, n_steps: int, seeds, use_mnist: bool = False, track_every: int = 0) -> dict:
    finals = {"teleport": [], "bounce": []}
    bests = []
    for s in seeds:
        r = run(cfg=replace(cfg, seed=s), n_steps=n_steps, use_mnist=use_mnist, track_every=track_every)
        finals["teleport"].append(r["error_probe_auc"]["teleport"])
        finals["bounce"].append(r["error_probe_auc"]["bounce"])
        if "best_error_probe_teleport" in r:
            bests.append(r["best_error_probe_teleport"]["auc"])
    agg = {"final_error_probe_teleport": _ms(finals["teleport"]),
           "final_error_probe_bounce": _ms(finals["bounce"])}
    if bests:
        agg["best_error_probe_teleport"] = _ms(bests)
    return agg
```
Update the `__main__` block to call `run(n_steps=10000, use_mnist=True, track_every=200)` (unchanged is fine). Update `notebooks/README.md` example if desired.

- [ ] **Step 4: Reconcile smoke tests**
Ensure the existing end-to-end smoke test still asserts only keys `run()` returns (it does — the v4 changes are additive: `loss_curve` entries gain probe keys, and `best_error_probe_teleport` is new). Keep the new `test_smoke_tracks_probe_over_training`. No removed keys.

- [ ] **Step 5: Run full suite + commit**
```bash
./.venv/Scripts/python.exe -m pytest -q
git add temporal_tf/run_prototype.py tests/test_smoke.py tests/test_run_seeds.py notebooks/README.md
git commit -m "feat: probe-over-training tracking, best-checkpoint, run_seeds aggregation"
```

---

## After this iteration runs (the decision)
Colab, stabilized + tracked + replicated:
```python
agg = run_seeds(
    Config(teleport_prob=0.1, batch_size=16, n_mask_draws=4, n_eval_clips=50,
           pred_coef=25, var_coef=25, cov_coef=1, lr=3e-4, grad_clip=1.0,
           warmup_steps=500, lr_schedule="cosine"),
    n_steps=10000, seeds=[0, 1, 2], use_mnist=True, track_every=200)
```
Read: (1) does stabilization keep `pred` low (no step-2400 spike) — inspect a single `run`'s `loss_curve`? (2) `best_error_probe_teleport.mean ± std` across seeds — is teleport decodable from error **consistently** > 0.5 and > the feature baseline? (3) does the probe AUC in the `loss_curve` **rise** as `pred` drops (signal tracks predictive quality)? If yes to all → the hypothesis holds robustly. If the probe stays ~0.5 even with stable low `pred` → genuine negative; revisit architecture (feature- vs error-forwarding, horizon, EMA target). Tuning frontier if needed: `cov` dominated v3's loss (~0.4 feature redundancy) — consider `cov_coef` adjustments.

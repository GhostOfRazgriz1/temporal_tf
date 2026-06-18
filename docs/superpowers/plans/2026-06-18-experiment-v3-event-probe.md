# Experiment v3 — Loss logging, localized surprise, teleport-detection probe

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development. Steps use `- [ ]`.

**Goal:** Stop guessing — instrument the prototype so we can decide the hypothesis cleanly. Add (1) loss-component logging, (2) localized surprise readouts (max / top-k over tokens, not just mean), and (3) a supervised **teleport-tick probe** on the per-tick prediction-ERROR features (with a refined-FEATURE probe as a baseline), reported as pooled AUC.

**Why:** Raw mean-over-tokens surprise washes out spatially-local events; and we cannot see whether the predictor is learning. The decisive question is: *given the processed sequence, is "when did the teleport happen" decodable from the prediction error?* — a stand-in for the downstream surprise-gated consumer.

**Tech Stack:** Python 3.13 venv at `./.venv`, PyTorch CPU, scikit-learn, pytest. Tests via `./.venv/Scripts/python.exe -m pytest`.

## Global Constraints
- **Backward compatibility:** existing public APIs unchanged except where listed; existing tests stay green. New `Config` field `surprise_topk: int = 4` defaults safely.
- **Clean test of surprise:** the headline probe consumes the prediction-ERROR (`target - P`) features, NOT raw refined features. A refined-feature probe is included only as a baseline to subtract.
- **Pooled AUC:** event AUCs pool all (score, label) pairs across clips into a single `roc_auc_score`, never per-clip averaging.
- **Determinism:** randomness via passed `torch.Generator`; probes use fixed `random_state`.
- Reductions derive from `Output.pairs` (each `(layer, tick, P, target)` with `P,target` shape `(B, n_tokens, d_model)`) — no model change needed. Layer 0 has no pairs → its surprise stays NaN.

---

### Task 1: Event-probe eval utilities

**Files:**
- Modify: `temporal_tf/config.py` (add `surprise_topk: int = 4`)
- Create: `temporal_tf/eval/event_probe.py`
- Test: `tests/test_event_probe.py`

**Interfaces (in `temporal_tf/eval/event_probe.py`):**
- `surprise_maps(out, n_layers, topk) -> dict{"mean","max","topk"}` — each `(B, T, n_layers)`, NaN where no pair; per-(layer,tick) reduction over tokens of the per-token error RMS `sqrt(mean_d (target-P)^2)`.
- `pooled_auc(scores: Tensor (M,), labels: Tensor (M,) bool) -> float` — drop NaN; return 0.5 if degenerate (one class) or constant scores; else `roc_auc_score`.
- `per_tick_error_features(out, n_layers, topk) -> Tensor (B, T, (n_layers-1)*3)` — per predictive layer, concat `[mean, max, topk]` of per-token error; NaN in rows where a layer has no pair at that tick.
- `event_probe(features: Tensor (N, F), labels: Tensor (N,) bool, test_size=0.3, seed=0) -> float` — drop rows containing NaN; if <2 classes after cleaning, return 0.5; else stratified `train_test_split` + `LogisticRegression(max_iter=1000)`, return pooled test-set AUC (`roc_auc_score` on predicted probabilities).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_event_probe.py
import torch
from temporal_tf.config import Config
from temporal_tf.model import TemporalDepthModel
from temporal_tf.eval.event_probe import surprise_maps, pooled_auc, per_tick_error_features, event_probe

def _cfg(): return Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=4,
                          clip_len=6, mask_ratio=0.5, surprise_topk=2)

def test_surprise_maps_shapes_and_layer0_nan():
    cfg = _cfg(); m = TemporalDepthModel(cfg)
    out = m(torch.rand(2, cfg.clip_len, 1, 16, 16), rng=torch.Generator().manual_seed(0))
    maps = surprise_maps(out, cfg.n_layers, cfg.surprise_topk)
    for r in ("mean", "max", "topk"):
        assert maps[r].shape == (2, cfg.clip_len, cfg.n_layers)
        assert torch.isnan(maps[r][:, :, 0]).all()                 # layer 0 never predicted
    # max >= mean wherever both defined
    defined = ~torch.isnan(maps["mean"])
    assert (maps["max"][defined] >= maps["mean"][defined] - 1e-6).all()

def test_pooled_auc():
    labels = torch.tensor([False, False, True, True])
    assert pooled_auc(torch.tensor([0.1, 0.2, 0.9, 0.8]), labels) > 0.9
    assert pooled_auc(torch.ones(4), labels) == 0.5                # constant
    assert pooled_auc(torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.tensor([True, True, True, True])) == 0.5  # degenerate

def test_per_tick_error_features_shape():
    cfg = _cfg(); m = TemporalDepthModel(cfg)
    out = m(torch.rand(2, cfg.clip_len, 1, 16, 16), rng=torch.Generator().manual_seed(0))
    feats = per_tick_error_features(out, cfg.n_layers, cfg.surprise_topk)
    assert feats.shape == (2, cfg.clip_len, (cfg.n_layers - 1) * 3)

def test_event_probe_separates():
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(60, 6, generator=g) + 2.5
    neg = torch.randn(60, 6, generator=g) - 2.5
    X = torch.cat([pos, neg]); y = torch.cat([torch.ones(60), torch.zeros(60)]).bool()
    assert event_probe(X, y) > 0.9
    assert abs(event_probe(torch.randn(120, 6, generator=g), y) - 0.5) < 0.2  # random ~ chance
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_event_probe.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add to `temporal_tf/config.py` (with the other v2 fields, before `n_tokens`): `surprise_topk: int = 4`.

Create `temporal_tf/eval/event_probe.py`:
```python
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def _token_err(P, target):
    # P, target: (B, N, d) -> per-token RMS error (B, N)
    return (target - P).pow(2).mean(-1).sqrt()

def surprise_maps(out, n_layers, topk):
    # find B, T from any pair (fallback to surprise tensor shape)
    B, T = out.surprise.shape[0], out.surprise.shape[1]
    maps = {r: torch.full((B, T, n_layers), float("nan")) for r in ("mean", "max", "topk")}
    for (li, t, P, target) in out.pairs:
        e = _token_err(P, target)                      # (B, N)
        k = min(topk, e.shape[-1])
        maps["mean"][:, t, li] = e.mean(-1)
        maps["max"][:, t, li] = e.max(-1).values
        maps["topk"][:, t, li] = e.topk(k, dim=-1).values.mean(-1)
    return maps

def pooled_auc(scores, labels):
    s = scores.detach().cpu().flatten()
    y = labels.detach().cpu().bool().flatten()
    valid = ~torch.isnan(s)
    s, y = s[valid], y[valid]
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    if torch.allclose(s, s[0].expand_as(s)):
        return 0.5
    return float(roc_auc_score(y.numpy(), s.numpy()))

def per_tick_error_features(out, n_layers, topk):
    B, T = out.surprise.shape[0], out.surprise.shape[1]
    n_pred = n_layers - 1
    feats = torch.full((B, T, n_pred * 3), float("nan"))
    for (li, t, P, target) in out.pairs:
        e = _token_err(P, target)                      # (B, N)
        k = min(topk, e.shape[-1])
        base = (li - 1) * 3
        feats[:, t, base + 0] = e.mean(-1)
        feats[:, t, base + 1] = e.max(-1).values
        feats[:, t, base + 2] = e.topk(k, dim=-1).values.mean(-1)
    return feats

def event_probe(features, labels, test_size=0.3, seed=0):
    X = features.detach().cpu().reshape(-1, features.shape[-1]).numpy()
    y = labels.detach().cpu().bool().reshape(-1).numpy()
    import numpy as np
    keep = ~np.isnan(X).any(1)
    X, y = X[keep], y[keep]
    if y.sum() < 2 or (~y).sum() < 2:
        return 0.5
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    if ytr.sum() == 0 or yte.sum() == 0:
        return 0.5
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
```

- [ ] **Step 4: Run tests (new + full suite)**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_event_probe.py -v` then `./.venv/Scripts/python.exe -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**
```bash
git add temporal_tf/config.py temporal_tf/eval/event_probe.py tests/test_event_probe.py
git commit -m "feat: event-probe eval utils (localized surprise + pooled AUC + supervised probe)"
```

---

### Task 2: Loss logging + instrumented run() + smoke

**Files:**
- Modify: `temporal_tf/train.py` (pass `train_step` parts to `on_eval`)
- Modify: `tests/test_train_v2.py` (callback now 3-arg)
- Modify: `temporal_tf/run_prototype.py` (instrumented `run` v3)
- Modify: `tests/test_smoke.py` (assert v3 keys)

**Interfaces:**
- `train(cfg, n_steps, digit_bank=None, device=None, on_eval=None, eval_every=0) -> model` — now calls `on_eval(model, step, parts)` (3 args; `parts` is `train_step`'s dict with `pred`/`var`/`cov`/`total`).
- `run(cfg=None, n_steps=300, use_mnist=False, track_every=0) -> dict` returns:
  - `loss_curve`: `list[{"step","pred","var","cov","total"}]` (from `on_eval`; empty if `track_every==0`),
  - `raw_pooled_auc`: `{"bounce":{"mean":[..per layer..],"max":[...],"topk":[...]}, "teleport":{...}}`,
  - `error_probe_auc`: `{"bounce":float,"teleport":float}` (supervised, pooled, from per-tick ERROR features),
  - `feature_probe_auc`: `{"bounce":float,"teleport":float}` (baseline, pooled refined features),
  - `collapse_std_per_layer`: `list[float]`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_smoke.py  (add; keep existing tests, but SEE step 5 about the v2 test's on_eval)
def test_end_to_end_smoke_v3():
    from temporal_tf.config import Config
    from temporal_tf.run_prototype import run
    cfg = Config(image_size=16, patch_size=8, d_model=24, n_heads=4, n_layers=4,
                 clip_len=6, mask_ratio=0.5, lr=3e-3, seed=0,
                 batch_size=2, teleport_prob=0.3, n_mask_draws=1, n_eval_clips=4, surprise_topk=2)
    report = run(cfg=cfg, n_steps=6, use_mnist=False, track_every=3)
    for k in ("loss_curve", "raw_pooled_auc", "error_probe_auc", "feature_probe_auc", "collapse_std_per_layer"):
        assert k in report
    assert len(report["loss_curve"]) == 2 and "pred" in report["loss_curve"][0]
    for ev in ("bounce", "teleport"):
        assert set(report["raw_pooled_auc"][ev]) == {"mean", "max", "topk"}
        assert len(report["raw_pooled_auc"][ev]["max"]) == cfg.n_layers - 1
        assert 0.0 <= report["error_probe_auc"][ev] <= 1.0
        assert 0.0 <= report["feature_probe_auc"][ev] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_smoke.py::test_end_to_end_smoke_v3 -v`
Expected: FAIL (run() lacks v3 keys).

- [ ] **Step 3: Implement**

In `temporal_tf/train.py`, change the callback call inside the loop from `on_eval(model, step)` to `on_eval(model, step, parts)` (where `parts = train_step(...)`).

In `tests/test_train_v2.py`, update the lambda to accept 3 args: `on_eval=lambda m, s, p: seen.append(s)`.

Rewrite `temporal_tf/run_prototype.py`:
```python
import torch
from temporal_tf.config import Config, default_config
from temporal_tf.train import train
from temporal_tf.data import synthetic_digit_bank, generate_batch
from temporal_tf.eval.collapse import representation_stats
from temporal_tf.eval.event_probe import surprise_maps, pooled_auc, per_tick_error_features, event_probe

def _collect(model, cfg, bank, n_clips):
    device = next(model.online.parameters()).device
    gen = torch.Generator().manual_seed(cfg.seed + 99)
    red = {"mean": [], "max": [], "topk": []}
    err_feats, feat_feats = [], []
    labels = {"bounce": [], "teleport": []}
    last = None
    with torch.no_grad():
        for _ in range(n_clips):
            clips, ev = generate_batch(gen, bank, cfg, B=1)
            out = model(clips.to(device), rng=torch.Generator().manual_seed(0))
            last = out
            maps = surprise_maps(out, cfg.n_layers, cfg.surprise_topk)
            for r in red:
                red[r].append(maps[r][0])                              # (T, n_layers)
            err_feats.append(per_tick_error_features(out, cfg.n_layers, cfg.surprise_topk)[0])  # (T, F)
            T = clips.shape[1]
            rows = []
            for t in range(T):
                vecs = [out.features[li][t][0].mean(0) for li in range(1, cfg.n_layers)]  # each (d,)
                rows.append(torch.cat(vecs))
            feat_feats.append(torch.stack(rows))                        # (T, (n_layers-1)*d)
            labels["bounce"].append(ev["bounce"][0])
            labels["teleport"].append(ev["teleport"][0])
    return red, err_feats, feat_feats, labels, last

def run(cfg: Config = None, n_steps: int = 300, use_mnist: bool = False, track_every: int = 0) -> dict:
    cfg = cfg or default_config()
    bank = None
    if use_mnist:
        from temporal_tf.data import load_mnist_digit_bank
        bank = load_mnist_digit_bank("./data")
    eval_bank = bank if bank is not None else synthetic_digit_bank(size=cfg.image_size // 2)

    loss_curve = []
    def on_eval(model, step, parts):
        loss_curve.append({"step": step, "pred": parts["pred"], "var": parts["var"],
                           "cov": parts["cov"], "total": parts["total"]})

    model = train(cfg, n_steps=n_steps, digit_bank=bank,
                  on_eval=(on_eval if track_every > 0 else None), eval_every=track_every)

    red, err_feats, feat_feats, labels, last = _collect(model, cfg, eval_bank, cfg.n_eval_clips)
    n_pred = cfg.n_layers - 1
    raw = {}
    for ev in ("bounce", "teleport"):
        y = torch.cat(labels[ev])                                       # (n_clips*T,)
        raw[ev] = {}
        for r in ("mean", "max", "topk"):
            scores = torch.cat(red[r])                                  # (n_clips*T, n_layers)
            raw[ev][r] = [pooled_auc(scores[:, li], y) for li in range(1, cfg.n_layers)]
    err_X = torch.cat(err_feats)                                        # (n_clips*T, n_pred*3)
    feat_X = torch.cat(feat_feats)
    error_probe = {ev: event_probe(err_X, torch.cat(labels[ev])) for ev in ("bounce", "teleport")}
    feature_probe = {ev: event_probe(feat_X, torch.cat(labels[ev])) for ev in ("bounce", "teleport")}
    collapse = [representation_stats(torch.stack(last.features[li]))["std"] for li in range(cfg.n_layers)]

    return {
        "loss_curve": loss_curve,
        "raw_pooled_auc": raw,
        "error_probe_auc": error_probe,
        "feature_probe_auc": feature_probe,
        "collapse_std_per_layer": collapse,
    }

if __name__ == "__main__":
    import json
    print(json.dumps(run(n_steps=10000, use_mnist=True, track_every=200), indent=2))
```
Update `notebooks/README.md`'s example call to `run(n_steps=10000, use_mnist=True, track_every=200)`.

- [ ] **Step 4: Reconcile the v2 smoke test**
The previous `test_end_to_end_smoke` and `test_end_to_end_smoke_v2` assert keys (`auc_bounce_per_layer`, `auc_teleport_per_layer`, `training_curve`, `cross_layer_surprise_corr`) that v3 `run()` no longer returns. UPDATE both: replace their key assertions with the v3 keys used in `test_end_to_end_smoke_v3` (or delete the now-redundant v2 test and keep only v3). Keep exactly one end-to-end smoke test asserting the v3 contract. Ensure the v2 `on_eval` test (`tests/test_train_v2.py`) uses the 3-arg callback.

- [ ] **Step 5: Run full suite + commit**
```bash
./.venv/Scripts/python.exe -m pytest -q
git add temporal_tf/train.py temporal_tf/run_prototype.py tests/test_train_v2.py tests/test_smoke.py notebooks/README.md
git commit -m "feat: instrumented run v3 (loss curve, pooled AUC, error + feature probes)"
```

---

## After this iteration runs (the decision)
Colab: `run(cfg=Config(teleport_prob=0.1, batch_size=16, n_mask_draws=4, n_eval_clips=50, pred_coef=25, var_coef=25, cov_coef=1), n_steps=10000, use_mnist=True, track_every=200)`. Read in order:
1. **`loss_curve`** — does `pred` actually drop? If not, the predictor isn't learning (training problem) — fix before judging.
2. **`error_probe_auc["teleport"]` vs `feature_probe_auc["teleport"]`** — the decisive comparison. error ≫ 0.5 and error > feature ⇒ surprise carries the event (hypothesis supported, raw eval was wrong readout). error ≈ 0.5 while pred drops ⇒ genuine negative ⇒ revisit architecture.
3. **`raw_pooled_auc[*]["max"]` vs `["mean"]`** — does max-over-tokens beat mean (confirms the localized-dilution diagnosis)?

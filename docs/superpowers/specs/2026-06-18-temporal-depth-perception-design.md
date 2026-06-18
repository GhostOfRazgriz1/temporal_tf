# Spec #1 — Temporal-Depth Predictive Perception Module

**Status:** Draft for review · **Date:** 2026-06-18 · **Program:** Predictive perception → surprise-gated memory

---

## 1. Motivation & vision

A standard transformer puts *time* on the sequence axis and *processing* on the depth axis. This architecture **folds time onto depth**: a streaming computation in which a frame's representation climbs one layer per tick, so network depth becomes an explicit, physical **temporal lag**.

The point of making time explicit is the **prediction-surprise signal**. Each layer predicts what it is about to receive and is corrected by reality; the error is *surprise*. Because depth = temporal lag, surprise is produced at a **hierarchy of timescales** — shallow layers fire on fast, fine changes; deep layers fire on slow, coarse **event boundaries**. That multi-timescale surprise is the deliverable: it is meant to gate a future **hippocampus-like episodic store** (Spec #2) that records representations when surprise is high.

This is **self-supervised**: there is no task head. The products are (a) refined per-frame representations and (b) the multi-timescale surprise signal.

### Grounding (why this is principled, not arbitrary)
- **Complementary Learning Systems** — slow statistical cortex (this hierarchy) + fast episodic hippocampus (the store).
- **Prediction-error-driven event segmentation** — Zacks et al.; Baldassano et al. 2017 (nested event timescales across cortex *and* hippocampus).
- **Nested temporal receptive windows** — Hasson et al. 2008: higher cortex integrates over longer time ⇒ depth = timescale.
- **Surprise-gated memory in ML** — Titans (Google, 2024) writes to neural memory gated on surprise.

The anti-aligned ordering (deeper = older) and the depth-induced latency are therefore **features**: the lag is the time the model needs to judge whether an old frame was surprising at a long timescale before committing it to memory.

---

## 2. Scope

This spec covers **only** the perception module.

- **In scope:** the streaming wavefront; the per-layer cell; the self-supervised predictive objective; anti-collapse machinery; and the **surprise + refined-feature interface** that a memory module would consume.
- **Out of scope (documented, not built):** the hippocampus / episodic store (Spec #2). We define its read interface so the module is "hippocampus-ready," nothing more.

---

## 3. Core mechanism

### 3.1 The wavefront

Frames `f_t` stream in at the bottom layer `L0`. Each frame's *refined feature* climbs **one layer per wall-clock tick**.

```
              t=0    t=1    t=2    t=3    t=4
 L0 (newest)  Z0⁰ →  Z0¹ →  Z0² →  Z0³ →  Z0⁴     ← raw frames f_t enter here
                ↘      ↘      ↘      ↘
 L1            ·     Z1⁰ →  Z1¹ →  Z1² →  Z1³
                       ↘      ↘      ↘
 L2            ·      ·     Z2⁰ →  Z2¹ →  Z2²
                              ↘      ↘
 Ln (oldest)  ·      ·      ·     Zn⁰ →  Zn¹
```

`Zₗᵏ` (layer `l`, frame `k`) is computed at wall-clock `t = k + l` from `Z₍ₗ₋₁₎ᵏ`. At any instant `t`, the live column is `[Z0ᵗ (newest) … Zₗᵗ⁻ˡ … Znᵗ⁻ⁿ (oldest)]`. **Deeper = older is a consequence of the pipeline, not a free parameter.**

### 3.2 Structural commitments

- **Untied across depth:** `L0 … Ln` have distinct parameters.
- **Recurrent across wall-clock time:** each layer is a recurrent predictive **cell** that carries temporal state across ticks (it reuses its own weights every tick).
- **Fixed stack depth `n`** is a hyperparameter = maximum temporal lag = number of distinct timescales. It is **independent of stream length**.

### 3.3 The cell: two outputs

Each layer `Lₗ`, on receiving the incoming refined feature `R₍ₗ₋₁₎` and updating its recurrent state `hₗ`, emits **two** things:

1. **Refined feature `Rₗ`** — passed **up**; the *only* thing `L₍ₗ₊₁₎` sees.
2. **Prediction `Pₗ`** — sent to a **loss head**, **not** passed up.

```
   incoming refined feature R_{l-1}  ──►┌──────────────────┐──►  R_l   (UP: forward path)
   recurrent state h_l (prev tick)  ──►│   Cell L_l        │
                                       │  (spatial attn +  │──►  P_l   (to loss head only)
                                       │   temporal recur) │
                                       └──────────────────┘──►  h_l (next tick)
```

- **Prediction target (horizon = 1, decision A):** `Pₗ` predicts **the next refined feature it will receive from below**, `R₍ₗ₋₁₎` one tick later. A growing-horizon variant (`Lₗ` predicts ~`l` ticks ahead) is a **switchable knob** (decision B), kept for the case where deep-layer surprise fails to decorrelate from shallow.
- **Firewall:** `Pₗ` is computed *before* the target arrives and never re-enters the forward path — so prediction cannot trivially copy its target.
- **`L0`** has no layer below it: it is a pure per-frame encoder (patch-embeds raw `f_t`), and prediction begins at `L1`.

### 3.4 Surprise

For each `(layer l, tick)`:
- **Error map** `eₗ = Rₜₐᵣ𝓰ₑₜ − Pₗ` (token-level, in latent space).
- **Scalar surprise** `sₗ = ‖eₗ‖` (a calibrated summary used for memory gating; exact reduction is an open detail, §7).

---

## 4. Training objective

Per layer `l ≥ 1`, at each tick:

```
L_pred(l) = distance( P_l ,  stop_grad( target_l ) )          # predict the future, in latent space
```

where `target_l` is the next-tick `R₍ₗ₋₁₎` produced by an **EMA (slowly-updated) copy** of the layer below (BYOL/V-JEPA style). Total loss = weighted sum over layers.

### Anti-collapse (mandatory — existential here)
A bare next-step latent predictor collapses to "predict next ≈ copy current" (adjacent frames are ~90–95% redundant), which would **null the surprise signal** and produce a dead memory gate. We bake in, from day one:

1. **Latent-space targets** (never pixel reconstruction) — CPC/JEPA rationale.
2. **Stop-gradient + EMA target** — prevents the target drifting to meet the prediction.
3. **Variance–covariance regularization (VICReg)** on each `Rₗ` — keeps representations spread and decorrelated; the explicit anti-collapse term (VJ-VCR lineage).
4. **Input masking (VideoMAE-style, 90–95%)** — makes each prediction non-trivial and discourages identity shortcuts.

### Training cost
The unrolled computation is 2D (depth × time) ⇒ backprop through time **and** depth. Mitigations: **truncated BPTT**, and — as a research option — **local per-layer losses** (predictive coding enables local credit assignment; not assumed to match backprop, treated as an experiment).

---

## 5. Surprise interface (hippocampus-ready)

The module exposes, per `(layer, tick)`, a record:

```
{ layer: l, frame_index: k, tick: t,
  refined_feature: R_l,        # candidate to store
  error_map:       e_l,
  surprise_scalar: s_l }       # gating signal
```

**Contract for a future store (Spec #2, not built):** read `s_l` across layers; when `s_l` exceeds a (possibly per-layer, adaptive) threshold, store `R_l` (and optionally the raw frame). Deeper layers ⇒ coarser events. The perception module makes **no** storage decisions itself.

---

## 6. Failure modes & mitigations

| Risk | Why it bites here | Mitigation |
|---|---|---|
| **Copy-collapse** (existential) | Kills the surprise signal → dead memory gate | Masking + latent targets + stop-grad/EMA + VICReg (§4) |
| **Depth doesn't differentiate** | Deep surprise ≈ shallow surprise ⇒ no timescale hierarchy | Measure shallow/deep surprise correlation; engage horizon-growth knob (B) |
| **Training cost** blows up | 2D unroll, untied params ∝ depth | Truncated BPTT; small `n`; optional local losses |
| **EMA / stop-grad instability** | Known in BYOL-style training | EMA momentum schedule; monitor representation rank/variance |
| **Stale deep features** | Deep layer describes frame `t−n` | Intended (retrospective encoding); document for downstream consumers |

---

## 7. Novelty & positioning

**Defensible novelty (verified by a grounding pass + a focused depth-as-time check):** mapping a *literal* temporal sequence onto depth with **one untied layer per tick**, doing **per-layer latent prediction**, and using **depth-as-explicit-lag to produce multi-timescale surprise** for memory gating. No surveyed system does this.

| Prior work | Shares | Differs |
|---|---|---|
| **PredNet** (Lotter et al. 2017, [1605.08104](https://arxiv.org/abs/1605.08104)) | Per-layer local prediction; recurrent video PC | Forwards **errors** up (we forward **features**); conv/ConvLSTM, **weight-tied** in time; depth = spatial hierarchy, not temporal lag |
| **Rao & Ballard 1999** ([Nat. Neuro.](https://www.nature.com/articles/nn0199_79)) | The PC bottleneck | Hierarchy = spatial scale; concurrent iterative relaxation, not a streaming depth=time wavefront |
| **JEPA / V-JEPA / VJ-VCR / CPC** ([2404.08471](https://arxiv.org/abs/2404.08471), [1807.03748](https://arxiv.org/pdf/1807.03748)) | Latent predictive SSL; no task head | Single encoder+predictor at **clip level**; no per-layer prediction, no time→depth |
| **Massively Parallel Video Networks** (Carreira et al. 2018, [1806.03863](https://arxiv.org/abs/1806.03863)) | Untied; deeper layers see **staler** frames (depth↔age) | Throughput trick on a **fixed-depth CNN** (same layers process every frame); no per-frame layer mapping; **no predictive objective** |
| **Depth-as-time lineage** (Neural ODEs, DEQ, Universal/Looped Transformers, LISTA) | Iteration/depth-as-time | Weight-**tied** (fail untied), or **single fixed input** iterated (fail per-frame input) |

---

## 8. Prototype plan (validates the theory; details in the implementation plan)

- **Data:** Moving MNIST first (Colab-friendly; bounces = genuine event/surprise points), then a second toy set (e.g., bouncing balls / KTH) if time allows.
- **Model size:** small — `n` ≈ 4–8 layers, `d_model` ≈ 128–256, small patches; runnable on a Colab T4/A100.
- **Core questions the prototype must answer:**
  1. **Does it learn without collapsing?** Prediction loss falls *and* representation variance/rank stays healthy (collapse monitor).
  2. **Is the surprise signal meaningful?** Per-tick `s_l` spikes at known events (bounces / appearance changes), low on smooth motion. Report localization precision/AUC.
  3. **Is there a timescale hierarchy?** Shallow vs deep surprise decorrelate; deep `s_l` aligns with coarser boundaries. (If not → engage knob B.)
  4. **Are the features useful?** Linear-probe `R_l` for digit identity / position / velocity.
- **Success criterion for "the architecture works":** a calibrated, multi-timescale surprise signal that localizes events, with non-collapsed, linearly-decodable features.

---

## 9. Open questions

1. **Horizon:** does depth-induced lag + recurrence alone yield timescale separation (A), or is growing horizon (B) needed? — empirical.
2. **Surprise reduction:** exact map → scalar (norm? learned? normalized by running baseline?).
3. **Recurrent state vs wavefront:** how far back does each cell's state integrate, and does that interact with the per-layer timescale?
4. **Local vs global training:** does per-layer local credit assignment hold up vs truncated BPTT?
5. **Anti-aligned vs conventional ordering:** ablate (the pipeline makes anti-aligned natural, but test it).

---

## 10. Operations

Git repo under **GhostOfRazgriz1**, developed locally and **run in Google Colab by cloning + pulling**. Concrete repo layout (`src/`, `notebooks/`, `configs/`, …) is defined in the implementation plan, not here.

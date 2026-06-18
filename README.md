# temporal_tf — Temporal-Depth Predictive Perception

A research prototype for a **predictive perception hierarchy that folds time onto network depth**.
A frame's representation climbs one layer per tick, so depth becomes an explicit temporal lag and
each layer emits a **prediction of the future**. The per-layer prediction error — *surprise* — is
produced at a hierarchy of timescales, intended to gate a future episodic ("hippocampus") store.

- **Self-supervised**, no task head. Deliverables: refined per-frame representations + a multi-timescale surprise signal.
- **Status:** design phase. See the spec before any code:
  [`docs/superpowers/specs/2026-06-18-temporal-depth-perception-design.md`](docs/superpowers/specs/2026-06-18-temporal-depth-perception-design.md)

## Running in Colab

This repo is developed locally and run in Google Colab by cloning and pulling:

```python
# in a Colab cell
!git clone https://github.com/GhostOfRazgriz1/temporal_tf.git
%cd temporal_tf
!git pull
```

(Training entrypoints and dependencies will be added during implementation.)

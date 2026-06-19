# Colab usage

```python
!git clone https://github.com/GhostOfRazgriz1/temporal_tf.git
%cd temporal_tf
%pip install -e .
from temporal_tf.run_prototype import run, run_seeds
from temporal_tf.config import Config

# Single run with probe-over-training tracking
report = run(n_steps=10000, use_mnist=True, track_every=200)
report

# Multi-seed aggregation
agg = run_seeds(
    Config(teleport_prob=0.1, batch_size=16, n_mask_draws=4, n_eval_clips=50,
           pred_coef=25, var_coef=25, cov_coef=1, lr=3e-4, grad_clip=1.0,
           warmup_steps=500, lr_schedule="cosine"),
    n_steps=10000, seeds=[0, 1, 2], use_mnist=True, track_every=200)
agg
```

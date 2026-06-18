# Colab usage

```python
!git clone https://github.com/GhostOfRazgriz1/temporal_tf.git
%cd temporal_tf
%pip install -e .
from temporal_tf.run_prototype import run
report = run(n_steps=500, use_mnist=True)
report
```

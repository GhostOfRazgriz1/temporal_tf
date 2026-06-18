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

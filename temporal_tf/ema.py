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

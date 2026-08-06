from __future__ import annotations
import torch
from torch import nn

def create_optimizer(model:nn.Module, learning_rate:float)->torch.optim.Optimizer:
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate
    )

    return optimizer
from __future__ import annotations
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu = nn.ReLU()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        return x
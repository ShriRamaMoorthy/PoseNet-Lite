from __future__ import annotations
import torch
import torch.nn as nn
from models.blocks import ConvBlock

class Backbone(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.output_channels=128
        self.stage1 = ConvBlock(in_channels=3,
                                out_channels=32)
        self.stage2 = ConvBlock(in_channels=32,
                                out_channels=64)
        self.stage3 = ConvBlock(in_channels=64,out_channels=128)

    def forward(self,x:torch.Tensor)->dict[str,torch.Tensor]:
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return {
            "stage1":x1,
            "stage2":x2,
            "stage3":x3
        }
    
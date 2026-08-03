from __future__ import annontations
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels:int,out_channels:int)->None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kerne_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.block(x)

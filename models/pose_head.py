from __future__ import annotations
import torch
import torch.nn as nn

class PoseHead(nn.Module):
    def __init__(self,in_channels:int,num_keypoints:int)->None:
        super().__init__()
        self.head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self,features:torch.Tensor)->torch.Tensor:
        return self.head(features)
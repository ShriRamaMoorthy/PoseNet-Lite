from __future__ import annotations
import torch
import torch.nn as nn

class HeatmapMSELoss(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')


    def forward(self,predictions:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
        return self.criterion(predictions,targets)
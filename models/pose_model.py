from __future__ import annotations
import torch
import torch.nn as nn
from models.backbone import Backbone
from models.pose_head import PoseHead




class PoseEstimationModel(nn.Module):
    def __init__(self, num_keypoints:int):
        super().__init__()
        self.backbone = Backbone()
        self.pose_head = PoseHead(
            in_channels=self.backbone.output_channels,
            num_keypoints=num_keypoints
        )

    def forward(self,images:torch.Tensor)->torch.Tensor:
        features = self.backbone(images)
        heatmaps = self.pose_head(
            features['stages3']
        )

        return heatmaps
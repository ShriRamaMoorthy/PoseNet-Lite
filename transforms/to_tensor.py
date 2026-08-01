from __future__ import annotations
import torch
from datasets.sample import PoseSample
from datasets.tensor_sample import TensorSample

class ToTensor:
    def __call__(self, sample:PoseSample)->PoseSample:
        image = torch.from_numpy(sample.image.transpose(2,0,1)).float()
        keypoints = torch.from_numpy(sample.keypoints).float()
        visibility = torch.from_numpy(sample.visibility).float()
        bbox = torch.from_numpy(sample.bbox).float()

        return TensorSample(
            image=image,
            keypoints=keypoints,
            visibility = visibility,
            bbox=bbox,
            image_id=sample.image_id
        )
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass(slots=True)
class TensorSample:
    image:torch.Tensor
    keypoints: torch.Tensor
    visibility: torch.Tensor
    bbox: torch.Tensor
    image_id: Optional[str]=None
    
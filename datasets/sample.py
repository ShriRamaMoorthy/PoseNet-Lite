from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(slots=True)
class PoseSample:
    image: np.ndarray
    keypoints:np.ndarray
    visibility:np.ndarray
    bbox:np.ndarray
    image_id:Optional[str]=None
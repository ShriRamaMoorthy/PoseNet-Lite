from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(slots=True)
class PoseAnnotation:
    image_path: Path
    keypoints: np.ndarray
    visibility:np.ndarray
    bbox:np.ndarray
    
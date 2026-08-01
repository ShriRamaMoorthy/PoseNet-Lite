from __future__ import annotations
import numpy as np
from datasets.sample import PoseSample

class Normalize:
    def __init__(self,mean:tuple[float,float,float],std:tuple[float,float,float])->None:
        self.mean = np.asarray(mean,dtype=np.float32)
        self.std = np.asarray(std,dtype=np.float32)

    def __call__(self, sample:PoseSample)->PoseSample:
        image = sample.image.astype(np.float32)
        image/=255.0
        image = (image-self.mean) / self.std
        return PoseSample(image=image, 
                          keypoints=sample.keypoints.copy(),
                          visibility = sample.visibility.copy(),
                          bbox = sample.bbox.copy(),
                          image_id=sample.image_id)
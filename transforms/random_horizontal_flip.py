from __future__ import annotations
import random
import cv2
import numpy as np
from datasets.sample import PoseSample

LEFT_RIGHT_PAIRS = (
    (0, 5),    # ankles
    (1, 4),    # knees
    (2, 3),    # hips
    (10, 15),  # wrists
    (11, 14),  # elbows
    (12, 13),  # shoulders
)

class RandomHorizontalFlip:
    """
    Flips image and all related annotations
    """
    def __init__(self,p:float=0.5)->None:
        self.p = p

    def __call__(self, sample:PoseSample)->PoseSample:
        if random.random() >= self.p:
            return sample

        image = cv2.flip(sample.image,1)

        width = image.shape[1]

        keypoints = sample.keypoints.copy()

        keypoints[:,0] = (width-1) - keypoints[:,0] # Flip x coordinates

        for left,right in LEFT_RIGHT_PAIRS:
            keypoints[[left,right]] = keypoints[[right,left]]  # Swap the joints

        visibility = sample.visibility.copy()

        # Keep visibility aligned with keypoints
        for left , right in LEFT_RIGHT_PAIRS:
            visibility[[left,right]] = visibility[[right,left]]

        bbox = sample.bbox.copy()

        xmin , ymin , xmax , ymax = bbox

        bbox[0] = (width-1) - xmax
        bbox[2] = (width-1) - xmin

        return PoseSample(image=image,
                          keypoints=keypoints,
                          visibility=visibility,
                          bbox=bbox,
                          image_id=sample.image_id)
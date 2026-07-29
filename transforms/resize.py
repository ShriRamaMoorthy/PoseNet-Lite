from __future__ import annotations
import cv2
import numpy as np
from datasets.sample import PoseSample

class Resize:
    def __init__(self,width:int,height:int)->None:
        self.width = width
        self.height = height

    def __call__(self, sample:PoseSample)->PoseSample:
        image = sample.image
        old_height, old_width = image.shape[:2]
        scale_x = self.width / old_width
        scale_y = self.height / old_height

        resized_image = cv2.resize(image,(self.width,self.height),interpolation=cv2.INTER_LINEAR)
        keypoints = sample.keypoints.copy()
        keypoints[:,0] *= scale_x
        keypoints[:,1] *= scale_y

        bbox = sample.bbox.copy()

        bbox[[0,2]] *= scale_x
        bbox[[1,3]] *= scale_y

        return PoseSample(image=resized_image,
                          keypoints=keypoints,
                          visibility=sample.visibility.copy(),
                          bbox=bbox,
                          image_id = sample.image_id)
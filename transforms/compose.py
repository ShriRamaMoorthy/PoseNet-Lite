from __future__ import annotations
from collections.abc import Iterable
from datasets.sample import PoseSample

class Compose:
    def __init__(self,transforms:Iterable):
        self.transforms = list(transforms)

    def __call__(self, sample:PoseSample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self):
        transform_names=", ".join(transform.__class__.__name__ for transform in self.transforms)
        return f'Compose([{transform_names}])'


### Testing
class AddOne:
    def __call__(self, value):
        return value+1

class Double:
    def __call__(self, value):
        return value*2
    
pipeline = Compose([
    AddOne(),
    Double()
])

print(pipeline(3))
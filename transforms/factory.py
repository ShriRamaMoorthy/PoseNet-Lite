from transforms.compose import Compose
from transforms.normalize import Normalize
from transforms.random_horizontal_flip import RandomHorizontalFlip
from transforms.resize import Resize
from transforms.to_tensor import ToTensor
from config.constants import (IMAGENET_HEIGHT,
                              IMAGENET_WIDTH,
                              IMAGENET_MEAN,
                              IMAGENET_STD,
                              HORIZONTAL_FLIP_PROBABILITY)

def build_train_transforms()->Compose:
    return Compose(
        [
            Resize(width=IMAGENET_WIDTH,
                   height=IMAGENET_HEIGHT),
            RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROBABILITY),
            Normalize(mean=IMAGENET_MEAN,
                      std=IMAGENET_STD),
            ToTensor(),
        ]
    )


def build_validation_transforms()->Compose:
    return Compose([
        Resize(width=IMAGENET_WIDTH,
               height=IMAGENET_HEIGHT),
        Normalize(mean=IMAGENET_MEAN,
                  std=IMAGENET_STD),
        ToTensor(),
    ])


def build_test_transforms()->Compose:
    return build_validation_transforms()
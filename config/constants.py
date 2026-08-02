# Used because we'll use a ResNet backbone pretrained on ImageNet
IMAGENET_MEAN = (0.485,0.456,0.406)
IMAGENET_STD = (0.229,0.224,0.225)

IMAGENET_WIDTH = 256
IMAGENET_HEIGHT = 256

HORIZONTAL_FLIP_PROBABILITY = 0.5

# Image Info
RGB_CHANNELS=3

MPII_KEYPOINT_NAMES = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "pelvis",
    "thorax",
    "upper_neck",
    "head_top",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
]


MPII_SKELETON = [
    (0, 1),
    (1, 2),
    (2, 6),
    (3, 6),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (8, 9),
    (7, 12),
    (12, 11),
    (11, 10),
    (7, 13),
    (13, 14),
    (14, 15),
]
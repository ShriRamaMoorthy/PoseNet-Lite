from dataclasses import dataclass,field

@dataclass
class DatasetConfig:
    image_size:int = 256
    heatmap_size:int = 64
    num_keypoints:int=16
    sigma:float=2.0


@dataclass
class TrainingConfig:
    batch_size:int = 16
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay:float = 1e-4


@dataclass
class SystemConfig:
    num_workers:int = 4
    pin_memory:bool=True
    seed:int=42


@dataclass
class ModelConfig:
    backbone:str="resnet18"
    pretrained:bool = True


@dataclass
class Config:
    dataset: DatasetConfig=field(default_factory=DatasetConfig)
    training: TrainingConfig=field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

config = Config()
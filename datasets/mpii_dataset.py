from pathlib import Path
from typing import Any
import cv2
from .base_dataset import BasePoseDataset


class MPIIDataset(BasePoseDataset):
    def __init__(self, root_dir:str|Path, annotations: list[dict[str,Any]],transforms:Any = None)->None:
        super().__init__(root_dir, transforms)
        self.annotations = annotations

    def __len__(self)->int:
        return len(self.annotations)

    def __getitem__(self, index:int)->dict[str,Any]:
        annotation = self.annotations[index]
        image = self._load_image(annotation)
        sample = {
            "image":image,
            "keypoints":annotation['keypoints'],
            "visibility":annotation['visibility'],
            "bbox":annotation['bbox']
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _load_image(self,annotation:dict[str,Any]):
        image_path = self.root_dir / annotation['image']
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Unable to load image: {image_path}")

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset

class BasePoseDataset(Dataset,ABC):
    def __init__(self,root_dir:str|Path,transforms:Any=None)->None:
        self.root_dir = Path(root_dir)
        self.transform = transforms


    @abstractmethod
    def __len__(self)->int:
        """
        Return the total number of samples.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self,index:int)->dict[str,Any]:
        raise NotImplementedError

    

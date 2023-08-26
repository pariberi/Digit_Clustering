from abc import ABC, abstractmethod

import numpy as np

from src.utils.transform_dataset import TransformDataset

import torch


class BaseModel(ABC):
    @abstractmethod
    def _load_model(self) -> tuple:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, test_sample: TransformDataset or np.ndarray) -> int:
        raise NotImplementedError()

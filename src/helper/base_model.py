from abc import ABC, abstractmethod

import torch


class BaseModel(ABC):
    @abstractmethod
    def _load_model(self) -> tuple:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X: torch.Tensor) -> int:
        raise NotImplementedError()

import numpy as np
import torch


class SimpleScale:
    @staticmethod
    def scale_tensor(dataset: torch.Tensor, number: float) -> torch.Tensor:
        return dataset.float().div_(number)

    @staticmethod
    def scale_ndarray(dataset: np.ndarray, number: float) -> np.ndarray:
        return np.divide(dataset, number)

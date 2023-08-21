import torch


class SimpleScale:
    @staticmethod
    def scale_tensor(dataset: torch.Tensor, number: float) -> torch.Tensor:
        return dataset.float().div_(number)

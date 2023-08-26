import numpy as np

from src.utils.file_loader import FileLoader
from src.utils.logger import Logger
from .base_model import BaseModel


class CNNClassification(BaseModel):
    __trained_cnn_model_path = 'D:\\Digit_Clustering\\src\\assets\\cnn\\model_saved.h5'

    @Logger.time_logger
    def __init__(self):
        print('Loading models...')
        self.model = self._load_model()

    def _load_model(self) -> tuple:
        return FileLoader.load_hdf5(CNNClassification.__trained_cnn_model_path)

    @Logger.time_logger
    def predict(self, test_sample: np.ndarray) -> int:
        print("\nStart of classification...")
        results = self.model.predict(test_sample)
        results2 = np.argmax(results, axis=1)

        return results2

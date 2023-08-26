import os
import pickle

import keras
import numpy as np
import torch
from PIL import Image, ImageOps
from keras.models import load_model
from torch import Tensor

from src.exception.image_exception import ImageException
from src.helper import nns


class FileLoader:
    @staticmethod
    def load_image_as_tensor(image_path: str) -> Tensor:
        # print(f'image path: {image_path}')
        if not os.path.exists(image_path):
            raise ImageException('path')
        if not image_path.endswith('.jpg') and not image_path.endswith('.png'):
            raise ImageException('format')

        img = Image.open(image_path)
        img = ImageOps.grayscale(img)
        img = np.array(img)
        img = np.expand_dims(img, axis=2)

        if img.shape[0] != 28 or img.shape[1] != 28:
            raise ImageException('size')

        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    @staticmethod
    def load_image_as_ndarray(image_path: str) -> np.ndarray:
        # print(f'image path: {image_path}')
        if not os.path.exists(image_path):
            raise ImageException('path')
        if not image_path.endswith('.jpg') and not image_path.endswith('.png'):
            raise ImageException('format')

        img = Image.open(image_path)
        img = ImageOps.grayscale(img)
        img = np.array(img)
        img = np.expand_dims(img, axis=2)

        if img.shape[0] != 28 or img.shape[1] != 28:
            raise ImageException('size')

        img = np.expand_dims(img, axis=0)
        return img

    @staticmethod
    def load_trained_aes(aes_dir_path: str) -> list:
        aes = []
        for aeid in range(5):
            path = f'{aes_dir_path}\\model_weights{aeid}.pth'
            # print('Loading trained aes parameter from', path)

            autoencoder = nns.make_ae(aeid, torch.device('cpu'), 50, 28, 1)
            autoencoder.load_state_dict(torch.load(path))

            aes.append(autoencoder)
        return aes

    @staticmethod
    def load_pickle(path: str) -> object:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        return data

    @staticmethod
    def load_hdf5(path: str) -> keras.engine.sequential.Sequential:
        model = load_model(path)
        return model

import os
import pickle

import torch
import numpy as np
from PIL import Image, ImageOps
from src.exception.image_exception import ImageException
from src.helper.neural_network import AutoEncoder


class FileLoader:
    @staticmethod
    def load_image_as_tensor(image_path: str) -> torch.Tensor:
        if not os.path.exists(image_path):
            raise ImageException('path')
        if image_path[-4:] != '.jpg' or image_path[-4:] != '.png':
            raise ImageException('format')

        img = Image.open(image_path)
        img = ImageOps.grayscale(img)
        img = np.array(img)
        img = np.expand_dims(img, axis=2)

        if img.shape[0] != 28 or img.shape[1]:
            raise ImageException('size')

        return torch.from_numpy(img)

    @staticmethod
    def load_trained_aes(aes_dir_path: str) -> list:
        aes = []
        for aeid in range(5):
            path = f'{aes_dir_path}/{aeid}.pt'
            print('Loading trained aes from', path)
            chkpt = torch.load(path, map_location=torch.device('cpu'))
            revived_ae = AutoEncoder(chkpt['enc'], chkpt['dec'], aeid)
            aes.append(revived_ae)
        return aes

    @staticmethod
    def load_pickle(path: str) -> object:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        return data

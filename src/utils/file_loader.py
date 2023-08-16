import os
import pickle

import torch
from torchvision.io import read_image

from src.exception.image_exception import ImageException
from src.helper.neural_network import AutoEncoder


class FileLoader:
    @staticmethod
    def load_image_as_tensor(image_path: str) -> torch.Tensor:

        if not os.path.exists(image_path):
            raise ImageException('path')
        if image_path[-4:] != '.jpg' or image_path[-4:] != '.png':
            raise ImageException('format')

        img = read_image(image_path)

        if len(list(img[0][0])) != 28:
            raise ImageException('size')

        return img

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
        file = open(path, 'rb')
        data = pickle.load(file)
        file.close()
        return data

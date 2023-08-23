import sys
import warnings

import numpy as np
import torch

from helper.cnn_classification import CNNClassification
from helper.selective_pseudo_label_clustering import SelectivePseudoLabelClustering
from src.exception.image_exception import ImageException
from utils.file_loader import FileLoader
from utils.normalizer import SimpleScale
from utils.transform_dataset import TransformDataset

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    torch.manual_seed(0)

    print('#########################################')
    print('#      Image Labeling with MNIST      #')
    print('#########################################')

    message = 'please enter the path of a 28*28 image for Labelling : '
    image_path = input(message)

    # image_path = 'C:\\Users\\p.berangi\\Desktop\\image\\3.jpg'

    message = 'Thank you...\nEnter 1 (for classification) or 2 (for clustering): '
    method_id = input(message)

    if method_id == '1':
        try:
            img: np.ndarray = FileLoader.load_image_as_ndarray(image_path)
        except ImageException as e:
            print(e)
            sys.exit(1)
        simple_scale = SimpleScale()
        img = simple_scale.scale_ndarray(img, 255.0)

        model = CNNClassification()
        label = model.predict(img)

    elif method_id == '2':
        try:
            img = FileLoader.load_image_as_tensor(image_path)
        except ImageException as e:
            print(e)
            sys.exit(1)

        simple_scale = SimpleScale()
        img = simple_scale.scale_tensor(img, 255.0)
        image_tuple = (torch.unsqueeze(img, 0), torch.unsqueeze(torch.Tensor(0), 0))
        image_dataset = TransformDataset(image_tuple, device=torch.device('cpu'))

        model = SelectivePseudoLabelClustering()
        label = model.predict(image_dataset)
    else:
        print('Only numbers 1 or 2 is acceptable...Please try again....')
        sys.exit(1)

    print(f'\nFinal label of your sample is {label}')

import sys
import warnings

import torch

from helper.selective_pseudo_label_clustering import SelectivePseudoLabelClustering
from src.exception.image_exception import ImageException
from utils.file_loader import FileLoader
from utils.normalizer import SimpleScale
from utils.transform_dataset import TransformDataset

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    torch.manual_seed(0)

    print('#########################################')
    print('#      Image Clustering with MNIST      #')
    print('#########################################')

    message = 'please enter the path of a 28*28 image to predict its content : '
    image_path = input(message)

    # image_path = 'D:\\Digit_Clustering\\src\\assets\\3.jpg'

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

    print(f'label of your sample is {label}')


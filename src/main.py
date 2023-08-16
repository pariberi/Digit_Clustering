import sys
import warnings
from time import time

from helper.selective_pseudo_label_clustering import SelectivePseudoLabelClustering
from src.exception.image_exception import ImageException
from utils.file_loader import FileLoader
from utils.normalizer import SimpleScale

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    print('#########################################')
    print('#      Image Clustering with MNIST      #')
    print('#########################################')

    global LOAD_START_TIME
    LOAD_START_TIME = time()

    message = 'please enter the path of an 28*28 image to predict its content : '
    image_path = input(message)

    try:
        img = FileLoader.load_image_as_tensor(image_path)
    except ImageException:
        sys.exit(1)

    X = SimpleScale.scale_tensor(img, 255.)

    trained_aes_path = '..\Data\Autoencoders'
    umap_path = '..\Data\Umaps'
    hdbscan_path = '..\Data\Hdbscan'

    model = SelectivePseudoLabelClustering(trained_aes_path, umap_path, hdbscan_path)
    label = model.get_label(X)

    print(f'label of your sample is {label}')

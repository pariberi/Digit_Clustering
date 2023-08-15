import sys
import warnings
from time import time

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    print('#########################################')
    print('#      Image Clustering with MNIST      #')
    print('#########################################')

    global LOAD_START_TIME
    LOAD_START_TIME = time()

    message = 'please enter the path of an 28*28 image to predict its content : '
    image_path = input(message)

import os

import numpy as np
import pandas as pd
import requests

# set options
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 200)


# D = 784  # number of input layers (aka number of pixels in the digit image)
# K = 10  # number of output layers (aka number of categories or number of digits)


def get_binarized_mnist_dataset(file_path, train_or_test_or_valid):
    print('Reading ' + train_or_test_or_valid + ' data...')
    df = pd.read_csv(file_path, delimiter=' ', header=None)
    X = df.values
    X = X.astype(np.float16)
    return X


def get_binarized_mnist_labels(file_path, train_or_test_or_valid):
    print('Reading ' + train_or_test_or_valid + ' labels...')
    with open(file_path) as f:
        y = np.array(f.readlines())
    return y.astype(np.int8)


# Downloads the dataset to `dir_path`.
def obtain(dir_path):
    binarized_mnist_base_url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/'

    binarized_mnist_train_filename = 'binarized_mnist_train.amat'
    binarized_mnist_test_filename = 'binarized_mnist_test.amat'
    binarized_mnist_valid_filename = 'binarized_mnist_valid.amat'

    if not os.path.exists(os.path.join(dir_path, binarized_mnist_train_filename)):
        print('Downloading Binarized MNIST train images file...')
        r = requests.get(binarized_mnist_base_url + binarized_mnist_train_filename)
        with open(os.path.join(dir_path, binarized_mnist_train_filename), 'wb') as f:
            f.write(r.content)
        print('[DONE]')

    if not os.path.exists(os.path.join(dir_path, binarized_mnist_test_filename)):
        print('Downloading Binarized MNIST test images file...')
        r = requests.get(binarized_mnist_base_url + binarized_mnist_test_filename)
        with open(os.path.join(dir_path, binarized_mnist_test_filename), 'wb') as f:
            f.write(r.content)
        print('[DONE]')

    if not os.path.exists(os.path.join(dir_path, binarized_mnist_valid_filename)):
        print('Downloading Binarized MNIST validation images file...')
        r = requests.get(binarized_mnist_base_url + binarized_mnist_valid_filename)
        with open(os.path.join(dir_path, binarized_mnist_valid_filename), 'wb') as f:
            f.write(r.content)
        print('[DONE]')

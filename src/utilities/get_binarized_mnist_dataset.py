import os
import sys

import numpy as np
import pandas as pd

if sys.version_info[0] == 2:
    from urllib import urlretrieve
elif sys.version_info[0] == 3:
    from urllib.request import urlretrieve

# set options
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 200)

D = 784  # number of input layers (or number of pixels in the digit image)
K = 10  # number of output layers (or number of categories or number of digits)


###############


def get_binarized_mnist_dataset(mnist_file, train_or_test_or_valid):
    print('Reading ' + train_or_test_or_valid + ' data...')
    df = pd.read_csv(mnist_file, delimiter=' ', header=None)
    X = df.values
    X = X.astype(np.float16)
    return X


def get_binarized_mnist_labels(mnist_labels_file, train_or_test_or_valid):
    print('Reading ' + train_or_test_or_valid + ' labels...')
    with open(mnist_labels_file) as f:
        y = np.array(f.readlines())
    return y.astype(np.int8)


def obtain(dir_path):
    '''
    Downloads the dataset to ``dir_path``.
    '''
    dir_path = os.path.expanduser(dir_path)
    print('Downloading the dataset')
    urlretrieve(
        'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat',
        os.path.join(dir_path, 'binarized_mnist_train.amat')
    )
    urlretrieve(
        'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat',
        os.path.join(dir_path, 'binarized_mnist_valid.amat')
    )
    urlretrieve(
        'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat',
        os.path.join(dir_path, 'binarized_mnist_test.amat')
    )
    print('Done                     ')

import os
import numpy as np
from keras.datasets import cifar10


def get_cifar10_dataset(cifar_path):
    # DOWNLOAD CIFAR-10 DATASET #
    fname = 'cifar-10-python.tar.gz'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    cache_subdir = ''
    cache_dir = cifar_path
    cifar10.get_file(fname=fname, origin=origin, cache_subdir=cache_subdir, extract=True, cache_dir=cache_dir)

    num_train_samples = 50000

    X_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    path = os.path.join(cache_dir, 'cifar-10-batches-py')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = cifar10.load_batch(fpath)
        X_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    X_test, y_test = cifar10.load_batch(fpath)

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    X_train = X_train.transpose((0, 2, 3, 1))
    X_test = X_test.transpose((0, 2, 3, 1))

    return (X_train, y_train), (X_test, y_test)

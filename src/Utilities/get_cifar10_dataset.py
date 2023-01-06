import os
import ssl
import tarfile

import numpy as np
from keras.datasets import cifar10

ssl._create_default_https_context = ssl._create_unverified_context


def get_cifar10_dataset(cifar_path):
    # DOWNLOAD CIFAR-10 DATASET #
    fname = 'cifar-10-python.tar.gz'
    origin = 'https://www.cs.toronto.edu/~kriz/' + fname
    compressed_file_path = os.path.join(cifar_path, fname)
    uncompressed_file_path = os.path.join(cifar_path, 'cifar-10-batches-py')

    uncompressed = True
    if not os.path.exists(uncompressed_file_path):
        uncompressed = False
    if uncompressed:
        for i in range(1, 6):
            fpath = os.path.join(uncompressed_file_path, 'data_batch_' + str(i))
            if not os.path.exists(fpath):
                uncompressed = False
    if not uncompressed:
        if not os.path.exists(compressed_file_path):
            cifar10.get_file(fname=fname, origin=origin, cache_subdir=cifar_path, extract=True)
        # open file
        compressed_file = tarfile.open(compressed_file_path)
        # extracting file
        compressed_file.extractall(path=cifar_path)

    num_train_samples = 50000

    X_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(uncompressed_file_path, 'data_batch_' + str(i))
        data, labels = cifar10.load_batch(fpath)
        X_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(uncompressed_file_path, 'test_batch')
    X_test, y_test = cifar10.load_batch(fpath)

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    X_train = X_train.transpose((0, 2, 3, 1))
    X_test = X_test.transpose((0, 2, 3, 1))

    return (X_train, y_train), (X_test, y_test)

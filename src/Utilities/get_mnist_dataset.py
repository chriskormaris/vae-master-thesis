import gzip
import os
import shutil

import requests
from mlxtend.data import loadlocal_mnist


def get_mnist_dataset(mnist_dataset_base_path):
    X_train_gz_filename = 'train-images-idx3-ubyte.gz'
    y_train_gz_filename = 'train-labels-idx1-ubyte.gz'
    X_test_gz_filename = 't10k-images-idx3-ubyte.gz'
    y_test_gz_filename = 't10k-labels-idx1-ubyte.gz'

    X_train_gz_path = mnist_dataset_base_path + X_train_gz_filename
    y_train_gz_path = mnist_dataset_base_path + y_train_gz_filename
    X_test_gz_path = mnist_dataset_base_path + X_test_gz_filename
    y_test_gz_path = mnist_dataset_base_path + y_test_gz_filename

    mnist_base_url = 'http://yann.lecun.com/exdb/mnist/'

    X_train_gz_url = mnist_base_url + X_train_gz_filename
    y_train_gz_url = mnist_base_url + y_train_gz_filename
    X_test_gz_url = mnist_base_url + X_test_gz_filename
    y_test_gz_url = mnist_base_url + y_test_gz_filename

    X_train_file_path = mnist_dataset_base_path + 'train-images.idx3-ubyte'
    y_train_file_path = mnist_dataset_base_path + 'train-labels.idx1-ubyte'
    X_test_file_path = mnist_dataset_base_path + 't10k-images.idx3-ubyte'
    y_test_file_path = mnist_dataset_base_path + 't10k-labels.idx1-ubyte'

    # Download files if they do not exist.
    if not os.path.exists(mnist_dataset_base_path):
        os.mkdir(mnist_dataset_base_path)

    if not os.path.exists(X_train_file_path):
        if not os.path.exists(X_train_gz_path):
            print('Downloading MNIST train images file...')
            r = requests.get(X_train_gz_url)
            with open(X_train_gz_path, 'wb') as f:
                f.write(r.content)
        with gzip.open(X_train_gz_path, 'rb') as f_in:
            with open(X_train_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    if not os.path.exists(y_train_file_path):
        if not os.path.exists(y_train_gz_path):
            print('Downloading MNIST train labels file...')
            r = requests.get(y_train_gz_url)
            with open(y_train_gz_path, 'wb') as f:
                f.write(r.content)
        with gzip.open(y_train_gz_path, 'rb') as f_in:
            with open(y_train_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    if not os.path.exists(X_test_file_path):
        if not os.path.exists(X_test_gz_path):
            print('Downloading MNIST test images file...')
            r = requests.get(X_test_gz_url)
            with open(X_test_gz_path, 'wb') as f:
                f.write(r.content)
        with gzip.open(X_test_gz_path, 'rb') as f_in:
            with open(X_test_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    if not os.path.exists(y_test_file_path):
        if not os.path.exists(y_test_gz_path):
            print('Downloading MNIST test labels file...')
            r = requests.get(y_test_gz_url)
            with open(y_test_gz_path, 'wb') as f:
                f.write(r.content)
        with gzip.open(y_test_gz_path, 'rb') as f_in:
            with open(y_test_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    X_train, y_train = loadlocal_mnist(images_path=X_train_file_path, labels_path=y_train_file_path)

    X_test, y_test = loadlocal_mnist(images_path=X_test_file_path, labels_path=y_test_file_path)

    os.remove(X_train_file_path)
    os.remove(y_train_file_path)
    os.remove(X_test_file_path)
    os.remove(y_test_file_path)

    return (X_train, y_train), (X_test, y_test)

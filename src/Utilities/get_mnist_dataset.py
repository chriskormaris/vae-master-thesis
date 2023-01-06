import gzip
import os
import shutil

import requests
from mlxtend.data import loadlocal_mnist


def get_mnist_dataset(mnist_dataset_base_path):
    X_train_gz_path = mnist_dataset_base_path + 'train-images-idx3-ubyte.gz'
    y_train_gz_path = mnist_dataset_base_path + 'train-labels-idx1-ubyte.gz'
    X_test_gz_path = mnist_dataset_base_path + 't10k-images-idx3-ubyte.gz'
    y_test_gz_path = mnist_dataset_base_path + 't10k-labels-idx1-ubyte.gz'

    # Download files if they do not exist.
    if not os.path.exists(X_train_gz_path) or not os.path.exists(y_train_gz_path) or \
        not os.path.exists(X_test_gz_path) or not os.path.exists(y_test_gz_path):
        if not os.path.exists(mnist_dataset_base_path):
            os.mkdir(mnist_dataset_base_path)

        print('Downloading MNIST files...')

        r1 = requests.get('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
        r2 = requests.get('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
        r3 = requests.get('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
        r4 = requests.get('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')

        with open(X_train_gz_path, 'wb') as f:
            f.write(r1.content)
        with open(y_train_gz_path, 'wb') as f:
            f.write(r2.content)
        with open(X_test_gz_path, 'wb') as f:
            f.write(r3.content)
        with open(y_test_gz_path, 'wb') as f:
            f.write(r4.content)

    X_train_file = mnist_dataset_base_path + 'train-images.idx3-ubyte'
    y_train_file = mnist_dataset_base_path + 'train-labels.idx1-ubyte'
    X_test_file = mnist_dataset_base_path + 't10k-images.idx3-ubyte'
    y_test_file = mnist_dataset_base_path + 't10k-labels.idx1-ubyte'

    with gzip.open(X_train_gz_path, 'rb') as f_in:
        with open(X_train_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    with gzip.open(y_train_gz_path, 'rb') as f_in:
        with open(y_train_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    with gzip.open(X_test_gz_path, 'rb') as f_in:
        with open(X_test_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    with gzip.open(y_test_gz_path, 'rb') as f_in:
        with open(y_test_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    X_train, y_train = loadlocal_mnist(images_path=X_train_file,labels_path=y_train_file)

    X_test, y_test = loadlocal_mnist(images_path=X_test_file,labels_path=y_test_file)

    os.remove(X_train_file)
    os.remove(y_train_file)
    os.remove(X_test_file)
    os.remove(y_test_file)

    return (X_train, y_train), (X_test, y_test)

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist as fashion_mnist_dataset
from keras.datasets import mnist as mnist_dataset

from src import *
from src.utilities import reduce_data, construct_missing_data, get_non_zero_percentage, rmse, mae
from src.utilities.knn_matrix_completion import kNNMatrixCompletion
from src.utilities.plot_utils import plot_images


def mnist(K=10, structured_or_random='structured', digits_or_fashion='digits'):
    missing_value = 0.5

    if digits_or_fashion == 'digits':
        output_images_path = output_img_base_path + 'knn_missing_values/mnist'
        mnist_data = mnist_dataset.load_data()
    else:
        output_images_path = output_img_base_path + 'knn_missing_values/fashion_mnist'
        mnist_data = fashion_mnist_dataset.load_data()

    (X_train, y_train), (X_test, y_test) = mnist_data

    # We will normalize all values between 0 and 1,
    # and we will flatten the 28x28 images into vectors of size 784.
    X_train = X_train / 255.
    X_test = X_test / 255.
    X_train = X_train.reshape((-1, np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((-1, np.prod(X_test.shape[1:])))

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    print()

    # build train data
    print('Building TRAIN data...')
    # reduce the number of train examples from 55000 to 10000
    X_train, y_train, _ = reduce_data(X_train, X_train.shape[0], 10000, y_train)

    # build test data
    print('Building TEST data...')
    # reduce the number of test examples from 10000 to 250
    X_test, y_test, _ = reduce_data(X_test, X_test.shape[0], 250, y_test)

    # construct data with missing values
    X_train_missing, _, _ = construct_missing_data(X_test, structured_or_random=structured_or_random)
    X_test_missing, X_test, y_test = construct_missing_data(X_test, y_test, structured_or_random=structured_or_random)

    # plot X_test data
    fig = plot_images(X_test, y_test, show_plot=False)
    fig.savefig(f'{output_images_path}/Test Data.png', bbox_inches='tight')
    plt.close()

    # plot test data with missing values
    fig = plot_images(X_test_missing, y_test, show_plot=False)
    fig.savefig(f'{output_images_path}/Test Data with Mixed Missing Values K={K}.png', bbox_inches='tight')
    plt.close()

    # Compute how sparse is the matrix X_train.
    # Print the percentage of non-missing entries compared to the total entries of the matrix.

    percentage = get_non_zero_percentage(X_test_missing)
    print(f'non missing values percentage in the TEST data: {percentage} %')

    print()

    # run K-NN
    print('Running %i-NN algorithm...' % K)
    print()

    start_time = time.time()
    X_test_predicted = kNNMatrixCompletion(X_train_missing, X_test_missing, K, missing_value)
    elapsed_time = time.time() - start_time

    print(f'k-nn predictions calculations time: {elapsed_time}')
    print()

    fig = plot_images(X_test_predicted, y_test, show_plot=False)
    fig.savefig(f'{output_images_path}/Predicted Test Data K={K}.png', bbox_inches='tight')
    plt.close()

    error1 = rmse(X_test, X_test_predicted)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_test, X_test_predicted)
    print(f'mean absolute error: {error2}')


if __name__ == '__main__':
    mnist()

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10 as cifar10_dataset

from src import *
from src.utilities import reduce_data, construct_missing_data, get_non_zero_percentage, rmse, mae
from src.utilities.knn_matrix_completion import kNNMatrixCompletion
from src.utilities.plot_utils import plot_images


def cifar10(K=10, structured_or_random='structured', rgb_or_grayscale='grayscale', category=3):
    missing_value = 0.5

    if rgb_or_grayscale.lower() == 'grayscale':
        output_images_path = output_img_base_path + 'knn_missing_values/cifar10_grayscale'
    else:
        output_images_path = output_img_base_path + 'knn_missing_values/cifar10_rgb'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    # LOAD CIFAR-10 DATASET #
    (X_train, y_train), (X_test, y_test) = cifar10_dataset.load_data()

    X_train = X_train[np.where(y_train == category)[0], :]
    X_test = X_test[np.where(y_test == category)[0], :]
    y_test = y_test[np.where(y_test == category)[0]]

    if rgb_or_grayscale.lower() == 'grayscale':
        # convert colored images from 3072 dimensions to 1024 grayscale images
        X_train = np.dot(X_train[:, :, :, :3], [0.299, 0.587, 0.114])
        X_train = np.reshape(X_train, newshape=(-1, 1024))  # X_train: N x 1024
        X_test = np.dot(X_test[:, :, :, :3], [0.299, 0.587, 0.114])
        X_test = np.reshape(X_test, newshape=(-1, 1024))  # X_test: N x 1024
    else:
        # We will flatten the 32x32 images into vectors of size 3072.
        X_train = X_train.reshape((-1, 3072))
        X_test = X_test.reshape((-1, 3072))

    # We will normalize all values between 0 and 1,
    X_train = X_train / 255.
    X_test = X_test / 255.

    # merge train and test data together to increase the train dataset size
    X_train = np.concatenate((X_train, X_test), axis=0)  # X_train: N x 3072

    # reduce the number of test examples to 100
    X_test, _, _ = reduce_data(X_test, X_test.shape[0], 100)

    # construct data with missing values
    X_train_missing, _, _ = construct_missing_data(X=X_train, structured_or_random=structured_or_random)
    X_test_missing, X_test, y_test = construct_missing_data(
        X=X_test,
        y=y_test,
        structured_or_random=structured_or_random
    )

    # plot test data
    fig = plot_images(
        X=X_test,
        y=y_test,
        categories=[category],
        n=100,
        grayscale=True if rgb_or_grayscale.lower() == 'grayscale' else False
    )
    fig.savefig(f'{output_images_path}/Test Data.png', bbox_inches='tight')
    plt.close()

    # plot original data with missing values
    fig = plot_images(
        X=X_test_missing,
        y=y_test,
        categories=[category],
        n=100,
        grayscale=True if rgb_or_grayscale.lower() == 'grayscale' else False
    )
    fig.savefig(f'{output_images_path}/Test Data with Mixed Missing Values K={K}', bbox_inches='tight')
    plt.close()

    # Compute how sparse is the matrix X_test_missing.
    # Print the percentage of non-missing entries compared to the total entries of the matrix.
    percentage = get_non_zero_percentage(X_test_missing)
    print(f'non missing values percentage in the TEST data: {percentage} %')

    print()

    # run K-NN
    print('Running %i-NN algorithm...' % K)
    print()

    start_time = time.time()
    # X_test_predicted = kNNMatrixCompletion(
    #     X_train_missing,
    #     X_test_missing,
    #     K,
    #     missing_value,
    #     use_softmax_weights=False
    # )
    X_test_predicted = kNNMatrixCompletion(X_train_missing, X_test_missing, K, missing_value)
    elapsed_time = time.time() - start_time

    print(f'k-nn predictions calculations time: {elapsed_time}')
    print()

    # plot predicted test data
    fig = plot_images(
        X=X_test_predicted,
        y=y_test,
        categories=[category],
        n=100,
        grayscale=True if rgb_or_grayscale.lower() == 'grayscale' else False
    )
    fig.savefig(f'{output_images_path}/Predicted Test Data K={K}', bbox_inches='tight')
    plt.close()

    error1 = rmse(X_test, X_test_predicted)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_test, X_test_predicted)
    print(f'mean absolute error: {error2}')


if __name__ == '__main__':
    cifar10()

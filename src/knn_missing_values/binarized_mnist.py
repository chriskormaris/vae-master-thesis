import os
import time

import matplotlib.pyplot as plt

from src import *
from src.utilities import reduce_data, construct_missing_data, get_non_zero_percentage, rmse, mae
from src.utilities.get_binarized_mnist_dataset import get_binarized_mnist_dataset, get_binarized_mnist_labels, obtain
from src.utilities.knn_matrix_completion import kNNMatrixCompletion
from src.utilities.plot_utils import plot_images


def binarized_mnist(K=10, structured_or_random='structured'):
    missing_value = 0.5

    output_images_path = output_img_base_path + 'knn_missing_values/binarized_mnist'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    if not os.path.exists(binarized_dataset_path):
        os.makedirs(binarized_dataset_path)
        obtain(binarized_dataset_path)

    print()

    # build train data
    # print('Building TRAIN data...')
    X_train = get_binarized_mnist_dataset(binarized_dataset_path + 'binarized_mnist_train.amat', 'TRAIN')
    # print(X_train.shape[0])
    # reduce the number of train examples from 50000 to 10000
    X_train, _, _ = reduce_data(X_train, X_train.shape[0], 10000)

    # build test data
    # print('Building TEST data...')
    X_test = get_binarized_mnist_dataset(binarized_dataset_path + 'binarized_mnist_test.amat', 'TEST')
    y_test = get_binarized_mnist_labels(binarized_dataset_path + 'binarized_mnist_test_labels.txt', 'TEST')
    # reduce the number of test examples from 10000 to 500
    X_test, y_test, _ = reduce_data(X_test, X_test.shape[0], 500, y=y_test)

    # construct data with missing values
    X_train_missing, _, _ = construct_missing_data(X_train, structured_or_random=structured_or_random)
    X_test_missing, X_test, y_test = construct_missing_data(X_test, y_test)

    # plot original data X_test
    fig = plot_images(X_test, y_test, show_plot=False)
    fig.savefig(f'{output_images_path}/Original Binarized Test Data.png', bbox_inches='tight')
    plt.close()

    # plot original data with missing values
    fig = plot_images(X_test_missing, y_test, show_plot=False)
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
    X_test_predicted = kNNMatrixCompletion(X_train_missing, X_test_missing, K, missing_value, binarize=True)
    elapsed_time = time.time() - start_time

    print(f'k-nn predictions calculations time: {elapsed_time}')
    print()

    fig = plot_images(X_test_predicted, y_test, show_plot=False)
    fig.savefig(f'{output_images_path}/Predicted Test Data K={K}', bbox_inches='tight')
    plt.close()

    error1 = rmse(X_test, X_test_predicted)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_test, X_test_predicted)
    print(f'mean absolute error: {error2}')


if __name__ == '__main__':
    binarized_mnist()

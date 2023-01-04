import inspect
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from __init__ import *


missing_value = 0.5
K = int(sys.argv[1])


###############

# MAIN #

if __name__ == '__main__':

    output_images_dir = './output_images/kNNMissingValues/binarized_mnist'
    binarized_mnist_dataset_dir = '../Binarized_MNIST_dataset'

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    if not os.path.exists(binarized_mnist_dataset_dir):
        os.makedirs(binarized_mnist_dataset_dir)
        get_binarized_mnist_dataset.obtain(binarized_mnist_dataset_dir)

    print('')

    num_classes = 10

    # build train data
    # print('Building TRAIN data...')
    X_train = get_binarized_mnist_dataset.get_binarized_mnist_dataset(binarized_mnist_dataset_dir + '/binarized_mnist_train.amat', 'TRAIN')
    print(X_train.shape[0])
    # reduce the number of train examples from 50000 to 10000
    X_train, _, _ = Utilities.reduce_data(X_train, X_train.shape[0], 10000)

    # construct data with missing values
    X_train_missing, X_train, _ = Utilities.construct_missing_data(X_train, structured_or_random=sys.argv[2])

    # build test data
    # print('Building TEST data...')
    X_test = get_binarized_mnist_dataset.get_binarized_mnist_dataset(binarized_mnist_dataset_dir + '/binarized_mnist_test.amat', 'TEST')
    print(X_test.shape[0])
    X_valid = get_binarized_mnist_dataset.get_binarized_mnist_dataset(binarized_mnist_dataset_dir + '/binarized_mnist_valid.amat', 'VALID')
    print(X_valid.shape[0])
    y_test = get_binarized_mnist_dataset.get_binarized_mnist_labels(binarized_mnist_dataset_dir + '/binarized_mnist_test_labels.txt', 'TEST')
    # reduce the number of test examples from 10000 to 500
    X_test, y_test, _ = Utilities.reduce_data(X_test, X_test.shape[0], 500, y=y_test)

    # construct data with missing values
    X_test_missing, X_test, y_test = Utilities.construct_missing_data(X_test, y_test)

    # plot original data X_test
    fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_test, y_test, show_plot=False)
    fig.savefig(output_images_dir + '/Original Binarized Test Data.png', bbox_inches='tight')
    plt.close()

    # plot original data with missing values
    fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_test_missing, y_test, show_plot=False)
    fig.savefig(output_images_dir + '/Test Data with Mixed Missing Values K=' + str(K), bbox_inches='tight')
    plt.close()

    # Compute how sparse is the matrix X_train. Print the percentage of non-missing entries compared to the total entries of the matrix.

    percentage = Utilities.non_missing_percentage(X_train_missing)
    print('non missing values percentage in the TRAIN data: ' + str(percentage) + ' %')
    percentage = Utilities.non_missing_percentage(X_test_missing)
    print('non missing values percentage in the TEST data: ' + str(percentage) + ' %')

    # convert variables to numpy matrices
    X_train = np.array(X_train)
    X_test_missing = np.array(X_test_missing)
    y_test = np.array(y_test)

    print('')

    # run K-NN
    print('Running %i-NN algorithm...' % K)
    print('')

    start_time = time.time()
    X_test_predicted = kNN.kNNMatrixCompletion(X_train_missing, X_test_missing, K, missing_value, binarize=True)
    # X_test_predicted = kNN.kNNMatrixCompletion(X_train_missing, X_test_missing, K, missing_value)
    elapsed_time = time.time() - start_time

    print('k-nn predictions calculations time: ' + str(elapsed_time))
    print('')

    fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_test_predicted, y_test, show_plot=False)
    fig.savefig(output_images_dir + '/Predicted Test Data K=' + str(K), bbox_inches='tight')
    plt.close()

    error1 = Utilities.rmse(X_test, X_test_predicted)
    print('root mean squared error: ' + str(error1))

    error2 = Utilities.mae(X_test, X_test_predicted)
    print('mean absolute error: ' + str(error2))

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


__author__ = 'c.kormaris'

missing_value = 0.5
K = int(sys.argv[1])


###############

# MAIN #

if __name__ == '__main__':

    cifar_dataset_dir = '../CIFAR_dataset/CIFAR-10'

    RGBOrGrayscale = sys.argv[3]

    if RGBOrGrayscale.lower() == 'grayscale':
        output_images_dir = './output_images/kNNMissingValues/cifar10_grayscale_cats_and_dogs'
    else:
        output_images_dir = './output_images/kNNMissingValues/cifar10_rgb_cats_and_dogs'

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    # LOAD CIFAR-10 DATASET #
    (X_train, y_train), (X_test, y_test) = get_cifar10_dataset.get_cifar10_dataset(cifar_dataset_dir, structured_or_random=sys.argv[2])

    if RGBOrGrayscale.lower() == 'grayscale':
        # convert colored images from 3072 dimensions to 1024 grayscale images
        X_train = np.dot(X_train[:, :, :, :3], [0.299, 0.587, 0.114])
        X_train = np.reshape(X_train, newshape=(-1, 1024))  # X_train: N x 1024
        X_test = np.dot(X_test[:, :, :, :3], [0.299, 0.587, 0.114])
        X_test = np.reshape(X_test, newshape=(-1, 1024))  # X_test: N x 1024
    else:
        # We will normalize all values between 0 and 1
        # and we will flatten the 32x32 images into vectors of size 3072.
        X_train = X_train.reshape((len(X_train), 3072))
        X_test = X_test.reshape((len(X_test), 3072))

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    # reduce train and test data to only two categories, the class 3 ('cat') and the class 5 ('dog')
    categories = [3, 5]

    category = 3
    X_train_cats = X_train[np.where(y_train == category)[0], :]
    y_train_cats = y_train[np.where(y_train == category)[0]]
    X_test_cats = X_test[np.where(y_test == category)[0], :]
    y_test_cats = y_test[np.where(y_test == category)[0]]

    category = 5
    X_train_dogs = X_train[np.where(y_train == category)[0], :]
    y_train_dogs = y_train[np.where(y_train == category)[0]]
    X_test_dogs = X_test[np.where(y_test == category)[0], :]
    y_test_dogs = y_test[np.where(y_test == category)[0]]

    X_train = np.concatenate((X_train_cats, X_train_dogs), axis=0)
    y_train = np.concatenate((y_train_cats, y_train_dogs), axis=0)
    X_test = np.concatenate((X_test_cats, X_test_dogs), axis=0)
    y_test = np.concatenate((y_test_cats, y_test_dogs), axis=0)

    # merge train and test data together to increase the train dataset size
    X_train = np.concatenate((X_train, X_test), axis=0)  # X_train: N x 3072
    y_train = np.concatenate((y_train, y_test), axis=0)

    # reduce the number of test examples to 100
    X_test, y_test, _ = Utilities.reduce_data(X_test, X_test.shape[0], 100, y=y_test)

    # construct data with missing values
    X_train_missing, X_train, y_train = Utilities.construct_missing_data(X_train, y_train)
    X_test_missing, X_test, y_test = Utilities.construct_missing_data(X_test, y_test)

    # plot train data
    if RGBOrGrayscale.lower() == 'grayscale':
        fig = plot_dataset_samples.plot_cifar10_data(X_train, y_train, categories=categories, n=50, grayscale=True)
    else:
        fig = plot_dataset_samples.plot_cifar10_data(X_train, y_train, categories=categories, n=50)
    fig.savefig(output_images_dir + '/Train Data.png', bbox_inches='tight')
    plt.close()

    # plot original data with missing values
    if RGBOrGrayscale.lower() == 'grayscale':
        fig = plot_dataset_samples.plot_cifar10_data(X_test_missing, y_test, categories=categories, n=50, grayscale=True)
    else:
        fig = plot_dataset_samples.plot_cifar10_data(X_test_missing, y_test, categories=categories, n=50)
    fig.savefig(output_images_dir + '/Test Data with Mixed Missing Values K=' + str(K), bbox_inches='tight')
    plt.close()

    # Compute how sparse is the matrix X_train. Print the percentage of non-missing entries compared to the total entries of the matrix.

    percentage = Utilities.non_missing_percentage(X_train_missing)
    print('non missing values percentage in the TRAIN data: ' + str(percentage) + ' %')
    percentage = Utilities.non_missing_percentage(X_test_missing)
    print('non missing values percentage in the TEST data: ' + str(percentage) + ' %')

    # convert variables to numpy matrices
    X_train = np.matrix(X_train)
    X_test_missing = np.matrix(X_test_missing)
    y_test = np.matrix(y_test).T

    print('')

    # run K-NN
    print('Running %i-NN algorithm...' % K)
    print('')

    start_time = time.time()
    #X_test_predicted = kNN.kNNMatrixCompletion(X_train_missing, X_test_missing, K, missing_value, use_softmax_weights=False)
    X_test_predicted = kNN.kNNMatrixCompletion(X_train_missing, X_test_missing, K, missing_value)
    elapsed_time = time.time() - start_time

    print('k-nn predictions calculations time: ' + str(elapsed_time))
    print('')

    # plot predicted test data
    if RGBOrGrayscale.lower() == 'grayscale':
        fig = plot_dataset_samples.plot_cifar10_data(X_test_predicted, y_test, categories=categories, n=50, grayscale=True)
    else:
        fig = plot_dataset_samples.plot_cifar10_data(X_test_predicted, y_test, categories=categories, n=50)
    fig.savefig(output_images_dir + '/Predicted Test Data K=' + str(K), bbox_inches='tight')
    plt.close()

    error1 = Utilities.rmse(X_test, X_test_predicted)
    print('root mean squared error: ' + str(error1))

    error2 = Utilities.mae(X_test, X_test_predicted)
    print('mean absolute error: ' + str(error2))

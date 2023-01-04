import inspect
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets.mnist import load_data

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from __init__ import *


missing_value = 0.5
K = int(sys.argv[1])


###############

# MAIN #

if __name__ == '__main__':

        digitsOrFashion = sys.argv[3]
        # digitsOrFashion = 'digits'
        # digitsOrFashion = 'fashion'

        if digitsOrFashion == 'digits':
                output_images_dir = './output_images/kNNMissingValues/mnist'
                mnist_dataset_dir = '../MNIST_dataset'
        else:
                output_images_dir = './output_images/kNNMissingValues/fashion_mnist'
                mnist_dataset_dir = '../FASHION_MNIST_dataset'

        mnist = load_data(mnist_dataset_dir, one_hot=True)

        X_train = mnist[0][0]
        X_train.reshape((-1, 784))
        y_train = mnist[0][1]
        t_train = np.zeros((y_train.size, 10))
        t_train[np.arange(y_train.size), y_train] = 1

        X_test = mnist[1][0]
        X_test.reshape((-1, 784))
        y_test = mnist[1][1]
        t_test = np.zeros((y_test.size, 10))
        t_test[np.arange(y_test.size), y_test] = 1

        if not os.path.exists(output_images_dir):
                os.makedirs(output_images_dir)

        print('')

        num_classes = 10

        # build train data
        print('Building TRAIN data...')
        y_train = np.argmax(t_train, axis=1)
        # reduce the number of train examples from 55000 to 10000
        X_train, y_train, _ = Utilities.reduce_data(X_train, X_train.shape[0], 10000, y_train)
        # X_train = binarize_data(X_train)

        # build test data
        print('Building TEST data...')
        y_test = np.argmax(t_test, axis=1)
        # reduce the number of test examples from 10000 to 250
        X_test, y_test, _ = Utilities.reduce_data(X_test, X_test.shape[0], 250, y_test)
        # X_test = binarize_data(X_test)

        # construct data with missing values
        X_train_missing, X_train, y_train = Utilities.construct_missing_data(X_train, y_train, structured_or_random=sys.argv[2])
        X_test_missing, X_test, y_test = Utilities.construct_missing_data(X_test, y_test, structured_or_random=sys.argv[2])

        # plot original data X_train
        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_train, y_train, show_plot=False)
        fig.savefig(output_images_dir + '/Train Data.png', bbox_inches='tight')
        plt.close()

        # plot X_test data
        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_test, y_test, show_plot=False)
        fig.savefig(output_images_dir + '/Test Data.png', bbox_inches='tight')
        plt.close()

        # plot data X_train_missing data
        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_train_missing, y_train, show_plot=False)
        fig.savefig(output_images_dir + '/Train Data with Mixed Missing Values.png', bbox_inches='tight')
        plt.close()

        # plot test data with missing values
        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_test_missing, y_test, show_plot=False)
        fig.savefig(output_images_dir + '/Test Data with Mixed Missing Values K=' + str(K) + '.png', bbox_inches='tight')
        plt.close()

        # Compute how sparse is the matrix X_train. Print the percentage of non-missing entries compared to the total entries of the matrix.

        percentage = Utilities.non_missing_percentage(X_train_missing)
        print('non missing values percentage in the TRAIN data: ' + str(percentage) + ' %')
        percentage = Utilities.non_missing_percentage(X_test_missing)
        print('non missing values percentage in the TEST data: ' + str(percentage) + ' %')

        # convert variables to numpy matrices
        X_train = np.array(X_train)
        X_test_missing = np.array(X_test_missing)
        y_test = np.ravel(np.array(y_test))

        print('')

        # run K-NN
        print('Running %i-NN algorithm...' % K)
        print('')

        start_time = time.time()
        # X_test_predicted = kNN.kNNMatrixCompletion(X_train_missing, X_test_missing, K, missing_value, binarize=True)
        X_test_predicted = kNN.kNNMatrixCompletion(X_train_missing, X_test_missing, K, missing_value)
        elapsed_time = time.time() - start_time

        print('k-nn predictions calculations time: ' + str(elapsed_time))
        print('')

        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_test_predicted, y_test, show_plot=False)
        fig.savefig(output_images_dir + '/Predicted Test Data K=' + str(K) + '.png', bbox_inches='tight')
        plt.close()

        error1 = Utilities.rmse(X_test, X_test_predicted)
        print('root mean squared error: ' + str(error1))

        error2 = Utilities.mae(X_test, X_test_predicted)
        print('mean absolute error: ' + str(error2))

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

    omniglot_dataset_dir = '../OMNIGLOT_dataset'

    language = RGBOrGrayscale = sys.argv[3]

    if language.lower() == 'greek':
        output_images_dir = './output_images/kNNMissingValues/omniglot_greek'
        alphabet = 20
    else:
        output_images_dir = './output_images/kNNMissingValues/omniglot_english'
        alphabet = 31

    if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)

    num_classes = 10

    # LOAD OMNIGLOT DATASET #
    X_train, y_train = get_omniglot_dataset.get_omniglot_dataset(omniglot_dataset_dir + '/chardata.mat', TrainOrTest='train', alphabet=alphabet, binarize=True)
    X_test, y_test = get_omniglot_dataset.get_omniglot_dataset(omniglot_dataset_dir + '/chardata.mat', TrainOrTest='test', alphabet=alphabet, binarize=True)

    X_merged = np.concatenate((X_train, X_test), axis=0)
    y_merged = np.concatenate((y_train, y_test), axis=0)

    #####

    # construct data with missing values
    X_merged_missing, X_merged, y_merged = Utilities.construct_missing_data(X_merged, y_merged, structured_or_random=sys.argv[2])

    # plot original data X_merged
    fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged, y_merged, categories=list(range(1, 11)), n=10, show_plot=False)
    fig.savefig(output_images_dir + '/Binarized Merged Data Characters 1-10.png', bbox_inches='tight')
    plt.close()
    fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged, y_merged, categories=list(range(11, 21)), n=10, show_plot=False)
    fig.savefig(output_images_dir + '/Binarized Merged Data Characters 11-20.png', bbox_inches='tight')
    plt.close()
    if language.lower() == 'greek':
        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged, y_merged, categories=list(range(21, 25)), n=10, show_plot=False)
        fig.savefig(output_images_dir + '/Binarized Merged Data Characters 21-24.png', bbox_inches='tight')
    else:
        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged, y_merged, categories=list(range(21, 27)), n=10, show_plot=False)
        fig.savefig(output_images_dir + '/Binarized Merged Data Characters 21-26.png', bbox_inches='tight')
    plt.close()

    # plot original data with missing values
    fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged_missing, y_merged, categories=list(range(1, 11)), n=10, show_plot=False)
    fig.savefig(output_images_dir + '/Merged Data with Mixed Missing Values K=' + str(K) + ' Characters 1-10.png', bbox_inches='tight')
    plt.close()
    fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged_missing, y_merged, categories=list(range(11, 21)), n=10, show_plot=False)
    fig.savefig(output_images_dir + '/Merged Data with Mixed Missing Values K=' + str(K) + ' Characters 11-20.png', bbox_inches='tight')
    plt.close()
    if language.lower() == 'greek':
        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged_missing, y_merged, categories=list(range(21, 25)), n=10, show_plot=False)
        fig.savefig(output_images_dir + '/Merged Data with Mixed Missing Values K=' + str(K) + ' Characters 21-24.png', bbox_inches='tight')
    else:
        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged_missing, y_merged, categories=list(range(21, 27)), n=10, show_plot=False)
        fig.savefig(output_images_dir + '/Merged Data with Mixed Missing Values K=' + str(K) + ' Characters 21-26.png', bbox_inches='tight')
    plt.close()

    # Compute how sparse is the matrix X_merged_missing. Print the percentage of non-missing entries compared to the total entries of the matrix.
    percentage = Utilities.non_missing_percentage(X_merged_missing)
    print('non missing values percentage: ' + str(percentage) + ' %')

    # convert variables to numpy matrices
    X_train = np.matrix(X_train)
    X_merged_missing = np.matrix(X_merged_missing)
    y_test = np.matrix(y_test).T

    print('')

    # run K-NN
    print('Running %i-NN algorithm...' % K)
    print('')

    start_time = time.time()
    X_merged_predicted = kNN.kNNMatrixCompletion(X_train, X_merged_missing, K, missing_value, binarize=True)
    # X_merged_predicted = kNN.kNNMatrixCompletion(X_train, X_merged_missing, K, missing_value)
    elapsed_time = time.time() - start_time

    print('k-nn predictions calculations time: ' + str(elapsed_time))
    print('')

    fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged_predicted, y_merged, categories=list(range(1, 11)), n=10, show_plot=False)
    fig.savefig(output_images_dir + '/Predicted Merged Data K=' + str(K) + ' Characters 1-10.png', bbox_inches='tight')
    plt.close()
    fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged_predicted, y_merged, categories=list(range(11, 21)), n=10, show_plot=False)
    fig.savefig(output_images_dir + '/Predicted Merged Data K=' + str(K) + ' Characters 11-20.png', bbox_inches='tight')
    plt.close()
    if language.lower() == 'greek':
        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged_predicted, y_merged, categories=list(range(21, 25)), n=10, show_plot=False)
        fig.savefig(output_images_dir + '/Predicted Merged Data K=' + str(K) + ' Characters 21-24.png', bbox_inches='tight')
    else:
        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged_predicted, y_merged, categories=list(range(21, 27)), n=10, show_plot=False)
        fig.savefig(output_images_dir + '/Predicted Merged Data K=' + str(K) + ' Characters 21-26.png', bbox_inches='tight')
    plt.close()

    error1 = Utilities.rmse(X_merged, X_merged_predicted)
    print('root mean squared error: ' + str(error1))

    error2 = Utilities.mae(X_merged, X_merged_predicted)
    print('mean absolute error: ' + str(error2))

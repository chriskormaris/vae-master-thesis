import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src.Utilities.constants import *
from src.Utilities.get_orl_faces_dataset import get_orl_faces_dataset
from src.Utilities.kNN_matrix_completion import kNNMatrixCompletion
from src.Utilities.plot_dataset_samples import plot_orl_faces
from src.Utilities.utils import construct_missing_data, get_non_zero_percentage, rmse, mae


def orl_faces(K=10, structured_or_random='structured'):
    missing_value = 0.5

    output_images_path = output_img_base_path + 'kNNMissingValues/orl_faces'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    # num_classes = 10

    # LOAD ORL FACES DATASET #
    X, y = get_orl_faces_dataset(orl_faces_dataset_path)

    # construct data with missing values
    X_missing, X, y = construct_missing_data(X, y, structured_or_random=structured_or_random)

    # plot original data X
    for i in range(0, 40, 10):
        fig = plot_orl_faces(X, y, categories=list(range(i, i + 10)), show_plot=False)
        fig.savefig(
            output_images_path + '/Original Faces ' + str(i + 1) + '-' + str(i + 10) + ' K=' + str(K) + '.png',
            bbox_inches='tight'
        )
        plt.close()

    # plot data with missing values
    for i in range(0, 40, 10):
        fig = plot_orl_faces(X_missing, y, categories=list(range(i, i + 10)), show_plot=False)
        fig.savefig(
            output_images_path + '/Faces ' + str(i + 1) + '-' + str(i + 10)
            + ' with Mixed Missing Values K=' + str(K) + '.png',
            bbox_inches='tight'
        )
        plt.close()

    # Compute how sparse is the matrix X_train.
    # Print the percentage of non-missing entries compared to the total entries of the matrix.
    percentage = get_non_zero_percentage(X_missing)
    print('non missing values percentage: ' + str(percentage) + ' %')

    # convert variables to numpy matrices
    X = np.array(X)
    X_missing = np.array(X_missing)

    print('')

    # run K-NN
    print('Running %i-NN algorithm...' % K)
    print('')

    start_time = time.time()
    X_predicted = kNNMatrixCompletion(X, X_missing, K, missing_value)
    elapsed_time = time.time() - start_time

    print('k-nn predictions calculations time: ' + str(elapsed_time))
    print('')

    # plot predicted data
    for i in range(0, 40, 10):
        fig = plot_orl_faces(X_predicted, y, categories=list(range(i, i + 10)), show_plot=False)
        fig.savefig(
            output_images_path + '/Predicted Faces ' + str(i + 1) + '-' + str(i + 10) + ' K=' + str(K) + '.png',
            bbox_inches='tight'
        )
        plt.close()

    error1 = rmse(X, X_predicted)
    print('root mean squared error: ' + str(error1))

    error2 = mae(X, X_predicted)
    print('mean absolute error: ' + str(error2))


if __name__ == '__main__':
    orl_faces()
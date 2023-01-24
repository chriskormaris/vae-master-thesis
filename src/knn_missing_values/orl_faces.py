import os
import time

import matplotlib.pyplot as plt

from src.utilities.constants import *
from src.utilities.get_orl_faces_dataset import get_orl_faces_dataset
from src.utilities.knn_matrix_completion import kNNMatrixCompletion
from src.utilities.plot_dataset_samples import plot_orl_faces
from src.utilities.utils import construct_missing_data, get_non_zero_percentage, rmse, mae


def orl_faces(K=10, structured_or_random='structured'):
    missing_value = 0.5

    output_images_path = output_img_base_path + 'knn_missing_values/orl_faces'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    # LOAD ORL FACES DATASET #
    X, y = get_orl_faces_dataset(orl_faces_dataset_path)

    # construct data with missing values
    X_missing, X, y = construct_missing_data(X, y, structured_or_random=structured_or_random)

    # plot original data X
    for i in range(0, 40, 10):
        fig = plot_orl_faces(X, y, categories=list(range(i, i + 10)), show_plot=False)
        fig.savefig(f'{output_images_path}/Original Faces {i + 1}-{i + 10} K={K}.png', bbox_inches='tight')
        plt.close()

    # plot data with missing values
    for i in range(0, 40, 10):
        fig = plot_orl_faces(X_missing, y, categories=list(range(i, i + 10)), show_plot=False)
        fig.savefig(
            f'{output_images_path}/Faces {i + 1}-{i + 10} with Mixed Missing Values K={K}.png',
            bbox_inches='tight'
        )
        plt.close()

    # Compute how sparse is the matrix X_train.
    # Print the percentage of non-missing entries compared to the total entries of the matrix.
    percentage = get_non_zero_percentage(X_missing)
    print(f'non missing values percentage: {percentage} %')

    print()

    # run K-NN
    print('Running %i-NN algorithm...' % K)
    print()

    start_time = time.time()
    X_predicted = kNNMatrixCompletion(X, X_missing, K, missing_value)
    elapsed_time = time.time() - start_time

    print(f'k-nn predictions calculations time: {elapsed_time}')
    print()

    # plot predicted data
    for i in range(0, 40, 10):
        fig = plot_orl_faces(X_predicted, y, categories=list(range(i, i + 10)), show_plot=False)
        fig.savefig(f'{output_images_path}/Predicted Faces {i + 1}-{i + 10} K={K}.png', bbox_inches='tight')
        plt.close()

    error1 = rmse(X, X_predicted)
    print(f'root mean squared error: {error1}')

    error2 = mae(X, X_predicted)
    print(f'mean absolute error: {error2}')


if __name__ == '__main__':
    orl_faces()

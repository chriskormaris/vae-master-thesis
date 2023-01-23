import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src.utilities.constants import *
from src.utilities.get_yale_faces_dataset import get_yale_faces_dataset
from src.utilities.knn_matrix_completion import kNNMatrixCompletion
from src.utilities.plot_dataset_samples import plot_yale_faces
from src.utilities.utils import construct_missing_data, get_non_zero_percentage, rmse, mae


def yale_faces(K=10, structured_or_random='structured'):
    missing_value = 0.5

    output_images_path = output_img_base_path + 'knn_missing_values/yale_faces'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    num_classes = 10

    # LOAD YALE FACES DATASET #
    print('Getting YALE faces dataset...')
    X, y = get_yale_faces_dataset(yale_faces_dataset_path)

    # construct data with missing values
    X_missing, X, y = construct_missing_data(X, y, structured_or_random=structured_or_random)

    # plot original data X
    fig = plot_yale_faces(X, y, categories=list(range(10)), show_plot=False)
    fig.savefig(
        output_images_path + '/Original Faces 1-10 K=' + str(K) + '.png',
        bbox_inches='tight'
    )
    plt.close()

    # plot data with missing values
    fig = plot_yale_faces(X_missing, y, categories=list(range(10)), show_plot=False)
    fig.savefig(
        output_images_path + '/Faces 1-10 with Mixed Missing Values K=' + str(K) + '.png',
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
    fig = plot_yale_faces(X_predicted, y, categories=list(range(10)), show_plot=False)
    fig.savefig(
        output_images_path + '/Predicted Faces 1-10 K=' + str(K) + '.png',
        bbox_inches='tight'
    )
    plt.close()

    error1 = rmse(X, X_predicted)
    print(f'root mean squared error: {error1}')

    error2 = mae(X, X_predicted)
    print(f'mean absolute error: {error2}')


if __name__ == '__main__':
    yale_faces()

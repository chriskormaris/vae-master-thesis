import os
import time

import matplotlib.pyplot as plt
import numpy as np

from Utilities.get_yale_faces_dataset import get_yale_faces_dataset
from Utilities.kNN_matrix_completion import kNNMatrixCompletion
from Utilities.plot_dataset_samples import plot_yale_faces
from Utilities.utils import construct_missing_data, get_non_zero_percentage, rmse, mae


def yale_faces(K=10, structured_or_random='structured'):
    missing_value = 0.5

    yale_faces_dataset_dir = '../YALE_dataset/CroppedYale'

    output_images_dir = './output_images/kNNMissingValues/yale_faces'

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    num_classes = 10

    # LOAD YALE FACES DATASET #
    print('Getting YALE faces dataset...')
    X, y = get_yale_faces_dataset(yale_faces_dataset_dir)

    # construct data with missing values
    X_missing, X, y = construct_missing_data(X, y, structured_or_random=structured_or_random)

    # plot original data X
    fig = plot_yale_faces(X, y, categories=list(range(10)), show_plot=False)
    fig.savefig(
        output_images_dir + '/Original Faces ' + str(1) + '-' + str(10) + ' K=' + str(K) + '.png',
        bbox_inches='tight'
    )
    plt.close()

    # plot data with missing values
    fig = plot_yale_faces(X_missing, y, categories=list(range(10)), show_plot=False)
    fig.savefig(
        output_images_dir + '/Faces ' + str(1) + '-' + str(10) + ' with Mixed Missing Values K=' + str(K) + '.png',
        bbox_inches='tight')
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
        output_images_dir + '/Predicted Faces ' + str(1) + '-' + str(10) + ' K=' + str(K) + '.png',
        bbox_inches='tight'
    )
    plt.close()

    error1 = rmse(X, X_predicted)
    print('root mean squared error: ' + str(error1))

    error2 = mae(X, X_predicted)
    print('mean absolute error: ' + str(error2))

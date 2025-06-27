import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src import *
from src.utilities import construct_missing_data, get_non_zero_percentage, rmse, mae
from src.utilities.get_omniglot_dataset import get_omniglot_dataset
from src.utilities.knn_matrix_completion import kNNMatrixCompletion
from src.utilities.plot_utils import plot_images


def omniglot(K=10, structured_or_random='structured', language='English'):
    missing_value = 0.5

    if language.lower() == 'greek':
        output_images_path = os.path.join(output_img_base_path, 'knn_missing_values', 'omniglot_greek')
        alphabet = 20
    else:
        output_images_path = os.path.join(output_img_base_path, 'knn_missing_values', 'omniglot_english')
        alphabet = 31

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    # LOAD OMNIGLOT DATASET #
    X_train, y_train = get_omniglot_dataset(
        os.path.join(omniglot_dataset_path, 'chardata.mat'),
        train_or_test='train',
        alphabet=alphabet,
        binarize=True
    )
    X_test, y_test = get_omniglot_dataset(
        os.path.join(omniglot_dataset_path, 'chardata.mat'),
        train_or_test='test',
        alphabet=alphabet,
        binarize=True
    )

    X_merged = np.concatenate((X_train, X_test), axis=0)
    y_merged = np.concatenate((y_train, y_test), axis=0)

    #####

    # construct data with missing values
    X_merged_missing, X_merged, y_merged = construct_missing_data(
        X_merged,
        y_merged,
        structured_or_random=structured_or_random
    )

    # plot original data X_merged
    fig = plot_images(
        X_merged,
        y_merged,
        categories=list(range(1, 11)),
        n=10,
        show_plot=False
    )
    fig.savefig(os.path.join(output_images_path, 'Binarized Merged Data Characters 1-10.png'), bbox_inches='tight')
    plt.close()
    fig = plot_images(
        X_merged,
        y_merged,
        categories=list(range(11, 21)),
        n=10,
        show_plot=False
    )
    fig.savefig(os.path.join(output_images_path, 'Binarized Merged Data Characters 11-20.png'), bbox_inches='tight')
    plt.close()
    if language.lower() == 'greek':
        fig = plot_images(
            X_merged,
            y_merged,
            categories=list(range(21, 25)),
            n=10,
            show_plot=False
        )
        fig.savefig(os.path.join(output_images_path, 'Binarized Merged Data Characters 21-24.png'), bbox_inches='tight')
    else:
        fig = plot_images(
            X_merged,
            y_merged,
            categories=list(range(21, 27)),
            n=10,
            show_plot=False
        )
        fig.savefig(os.path.join(output_images_path, 'Binarized Merged Data Characters 21-26.png'), bbox_inches='tight')
    plt.close()

    # plot original data with missing values
    fig = plot_images(
        X_merged_missing,
        y_merged,
        categories=list(range(1, 11)),
        n=10,
        show_plot=False
    )
    fig.savefig(
        os.path.join(output_images_path, f'Merged Data with Missing Values K={K} Characters 1-10.png'),
        bbox_inches='tight'
    )
    plt.close()
    fig = plot_images(
        X_merged_missing,
        y_merged,
        categories=list(range(11, 21)),
        n=10,
        show_plot=False
    )
    fig.savefig(
        os.path.join(output_images_path, f'Merged Data with Missing Values K={K} Characters 11-20.png'),
        bbox_inches='tight'
    )
    plt.close()
    if language.lower() == 'greek':
        fig = plot_images(
            X_merged_missing,
            y_merged,
            categories=list(range(21, 25)),
            n=10,
            show_plot=False
        )
        fig.savefig(
            os.path.join(output_images_path, f'Merged Data with Missing Values K={K} Characters 21-24.png'),
            bbox_inches='tight'
        )
    else:
        fig = plot_images(
            X_merged_missing,
            y_merged,
            categories=list(range(21, 27)),
            n=10,
            show_plot=False
        )
        fig.savefig(
            os.path.join(output_images_path, f'Merged Data with Missing Values K={K} Characters 21-26.png'),
            bbox_inches='tight'
        )
    plt.close()

    # Compute how sparse is the matrix X_merged_missing.
    # Print the percentage of non-missing entries compared to the total entries of the matrix.
    percentage = get_non_zero_percentage(X_merged_missing)
    print(f'non missing values percentage: {percentage} %')

    print()

    # run K-NN
    print('Running %i-NN algorithm...' % K)
    print()

    start_time = time.time()
    X_merged_predicted = kNNMatrixCompletion(X_train, X_merged_missing, K, missing_value, binarize=True)
    elapsed_time = time.time() - start_time

    print(f'k-nn predictions calculations time: {elapsed_time}')
    print()

    fig = plot_images(
        X_merged_predicted,
        y_merged,
        categories=list(range(1, 11)),
        n=10,
        show_plot=False
    )
    fig.savefig(
        os.path.join(output_images_path, f'Predicted Merged Data K={K} Characters 1-10.png'),
        bbox_inches='tight'
    )
    plt.close()
    fig = plot_images(
        X_merged_predicted,
        y_merged,
        categories=list(range(11, 21)),
        n=10,
        show_plot=False
    )
    fig.savefig(
        os.path.join(output_images_path, f'Predicted Merged Data K={K} Characters 11-20.png'),
        bbox_inches='tight'
    )
    plt.close()
    if language.lower() == 'greek':
        fig = plot_images(
            X_merged_predicted,
            y_merged,
            categories=list(range(21, 25)),
            n=10,
            show_plot=False
        )
        fig.savefig(
            os.path.join(output_images_path, f'Predicted Merged Data K={K} Characters 21-24.png'),
            bbox_inches='tight'
        )
    else:
        fig = plot_images(
            X_merged_predicted,
            y_merged,
            categories=list(range(21, 27)),
            n=10,
            show_plot=False
        )
        fig.savefig(
            os.path.join(output_images_path, f'Predicted Merged Data K={K} Characters 21-26.png'),
            bbox_inches='tight'
        )
    plt.close()

    error1 = rmse(X_merged, X_merged_predicted)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_merged, X_merged_predicted)
    print(f'mean absolute error: {error2}')


if __name__ == '__main__':
    omniglot()

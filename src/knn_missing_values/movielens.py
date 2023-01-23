import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utilities.constants import *
from src.utilities.get_movielens_dataset import get_movielens_dataset
from src.utilities.knn_matrix_completion import kNNMatrixCompletion
from src.utilities.plot_dataset_samples import plot_movielens_data
from src.utilities.utils import get_non_zero_percentage, rmse, mae


def movielens(K=10):
    missing_value = 0

    output_data_path = movielens_output_data_base_path + 'knn_missing_values/'

    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)

    X_train, _, X_merged = get_movielens_dataset(movielens_dataset_path)

    # Compute how sparse is the matrix X_train.
    # Print the percentage of non-missing entries compared to the total entries of the matrix.

    percentage = get_non_zero_percentage(X_train)
    print(f'non missing values percentage in the TRAIN data: {percentage} %')

    num_users = X_train.shape[0]
    num_movies = X_train.shape[1]

    print(f'number of users: {num_users}')
    print(f'number of movies: {num_movies}')

    X_df = pd.DataFrame(X_train)
    X_df = X_df.replace(to_replace=0.0, value='---')
    X_df.to_csv(
        path_or_buf=output_data_path + '/users_movies_ratings_missing_values.csv',
        sep='\t',
        index=False,
        header=False
    )

    print()

    # run K-NN
    print('Running %i-NN algorithm...' % K)
    print()

    start_time = time.time()
    X_merged_predicted = kNNMatrixCompletion(X_train, X_merged, K, missing_value, binarize=True)
    elapsed_time = time.time() - start_time

    print(f'k-nn predictions calculations time: {elapsed_time}')
    print()

    X_merged_predicted[np.where(X_merged_predicted == missing_value)] = 1

    error1 = rmse(X_merged, X_merged_predicted)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_merged, X_merged_predicted)
    print(f'mean absolute error: {error2}')

    X_merged_predicted_df = pd.DataFrame(X_merged_predicted)
    # X_merged_predicted_df = X_merged_predicted_df.replace(to_replace=0.0, value='---')
    X_merged_predicted_df.to_csv(
        path_or_buf=output_data_path + '/users_movies_ratings_predicted_values.csv',
        sep='\t',
        index=False,
        header=False
    )

    fig = plot_movielens_data(X_merged_predicted)
    fig.savefig(output_data_path + '/average_movies_ratings.png', bbox_inches='tight')
    plt.close()

    X_merged_predicted[np.where(X_merged == 0)] = 0
    X_merged_predicted = np.round(X_merged_predicted)
    accuracy = np.not_equal(X_merged, X_merged_predicted).size / X_merged.size
    print(f'accuracy: {accuracy}')


if __name__ == '__main__':
    movielens()

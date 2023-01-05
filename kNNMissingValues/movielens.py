import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Utilities.get_movielens_dataset import get_movielens_dataset
from Utilities.kNN_matrix_completion import kNNMatrixCompletion
from Utilities.plot_dataset_samples import plot_movielens_data
from Utilities.utils import get_non_zero_percentage, rmse, mae


def movielens(K=10):
    missing_value = 0

    movielens_dataset_dir = '../MovieLens_dataset/ml-100k'
    output_data_dir = './movielens_output_data/kNNMissingValues/'

    if not os.path.exists(output_data_dir):
        os.mkdir(output_data_dir)

    X_train, _, X_merged = get_movielens_dataset(movielens_dataset_dir)

    # Compute how sparse is the matrix X_train.
    # Print the percentage of non-missing entries compared to the total entries of the matrix.

    percentage = get_non_zero_percentage(X_train)
    print('non missing values percentage in the TRAIN data: ' + str(percentage) + ' %')

    no_users = X_train.shape[0]
    no_movies = X_train.shape[1]

    print('number of users: ' + str(no_users))
    print('number of movies: ' + str(no_movies))

    X_df = pd.DataFrame(X_train)
    X_df = X_df.replace(to_replace=0.0, value='---')
    X_df.to_csv(
        path_or_buf=output_data_dir + '/users_movies_ratings_missing_values.csv',
        sep='\t',
        index=False,
        header=False
    )

    print('')

    # run K-NN
    print('Running %i-NN algorithm...' % K)
    print('')

    start_time = time.time()
    X_merged_predicted = kNNMatrixCompletion(X_train, X_merged, K, missing_value, binarize=True)
    elapsed_time = time.time() - start_time

    print('k-nn predictions calculations time: ' + str(elapsed_time))
    print('')

    X_merged_predicted[np.where(X_merged_predicted == missing_value)] = 1

    error1 = rmse(X_merged, X_merged_predicted)
    print('root mean squared error: ' + str(error1))

    error2 = mae(X_merged, X_merged_predicted)
    print('mean absolute error: ' + str(error2))

    X_merged_predicted_df = pd.DataFrame(X_merged_predicted)
    # X_merged_predicted_df = X_merged_predicted_df.replace(to_replace=0.0, value='---')
    X_merged_predicted_df.to_csv(
        path_or_buf=output_data_dir + '/users_movies_ratings_predicted_values.csv',
        sep='\t',
        index=False,
        header=False
    )

    fig = plot_movielens_data(X_merged_predicted)
    fig.savefig(output_data_dir + '/average_movies_ratings.png', bbox_inches='tight')
    plt.close()

    X_merged_predicted[np.where(X_merged == 0)] = 0
    X_merged_predicted = np.round(X_merged_predicted)
    accuracy = np.not_equal(X_merged, X_merged_predicted).size / X_merged.size
    print('accuracy: ' + str(accuracy))

import inspect
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from __init__ import *


missing_value = 0
K = int(sys.argv[1])


###############

# MAIN #

if __name__ == '__main__':

    movielens_dataset_dir = '../MovieLens_dataset/ml-100k'
    output_data_dir = './movielens_output_data/kNNMissingValues/'

    if not os.path.exists(output_data_dir):
        os.mkdir(output_data_dir)

    X_train, _, X_merged = get_movielens_dataset.get_movielens_data(movielens_dataset_dir)

    # Compute how sparse is the matrix X_train. Print the percentage of non-missing entries compared to the total entries of the matrix.

    percentage = Utilities.non_zero_percentage(X_train)
    print('non missing values percentage in the TRAIN data: ' + str(percentage) + ' %')

    no_users = X_train.shape[0]
    no_movies = X_train.shape[1]

    print('number of users: ' + str(no_users))
    print('number of movies: ' + str(no_movies))

    X_df = pd.DataFrame(X_train)
    X_df = X_df.replace(to_replace=0.0, value='---')
    X_df.to_csv(path_or_buf=output_data_dir + '/users_movies_ratings_missing_values.csv', sep='\t', index=None, header=None)

    print('')

    # run K-NN
    print('Running %i-NN algorithm...' % K)
    print('')

    start_time = time.time()
    X_merged_predicted = kNN.kNNMatrixCompletion(X_train, X_merged, K, missing_value, binarize=True)
    elapsed_time = time.time() - start_time

    print('k-nn predictions calculations time: ' + str(elapsed_time))
    print('')

    X_merged_predicted[np.where(X_merged_predicted == missing_value)] = 1

    error1 = Utilities.rmse(X_merged, X_merged_predicted)
    print('root mean squared error: ' + str(error1))

    error2 = Utilities.mae(X_merged, X_merged_predicted)
    print('mean absolute error: ' + str(error2))

    X_merged_predicted_df = pd.DataFrame(X_merged_predicted)
    # X_merged_predicted_df = X_merged_predicted_df.replace(to_replace=0.0, value='---')
    X_merged_predicted_df.to_csv(path_or_buf=output_data_dir + '/users_movies_ratings_predicted_values.csv', sep='\t', index=None, header=None)

    fig = plot_dataset_samples.plot_movielens_data(X_merged_predicted)
    fig.savefig(output_data_dir + '/average_movies_ratings.png', bbox_inches='tight')
    plt.close()

    X_merged_predicted[np.where(X_merged == 0)] = 0
    X_merged_predicted = np.round(X_merged_predicted)
    accuracy = np.not_equal(X_merged, X_merged_predicted).size / X_merged.size
    print('accuracy: ' + str(accuracy))

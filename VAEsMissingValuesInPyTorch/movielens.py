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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings

missing_value = 0

#####

# MAIN #

if __name__ == '__main__':

    movielens_dataset_dir = '../MovieLens_dataset/ml-100k'
    output_data_dir = './movielens_output_data/VAEsMissingValuesInPyTorch/'

    if not os.path.exists(output_data_dir):
        os.mkdir(output_data_dir)

    _, _, X_merged = get_movielens_dataset.get_movielens_data(movielens_dataset_dir)

    no_users = X_merged.shape[0]
    no_movies = X_merged.shape[1]

    print('number of users: ' + str(no_users))
    print('number of movies: ' + str(no_movies))

    X_df = pd.DataFrame(X_merged)
    X_df = X_df.replace(to_replace=0.0, value='---')
    X_df.to_csv(path_or_buf=output_data_dir + '/users_movies_ratings_missing_values.csv',
                sep='\t', index=None, header=None)

    print('')

    #####

    N = no_users
    input_dim = no_movies  # D
    # M1: number of neurons in the encoder
    # M2: number of neurons in the decoder
    hidden_encoder_dim = 400  # M1
    hidden_decoder_dim = hidden_encoder_dim  # M2
    latent_dim = int(sys.argv[1])  # Z_dim
    epochs = int(sys.argv[2])
    batch_size = sys.argv[3]
    if batch_size == 'N':
        batch_size = N
    else:
        batch_size = int(batch_size)
    learning_rate = float(sys.argv[4])

    #####

    params, solver = initialize_weights_in_pytorch.initialize_weights(input_dim, hidden_encoder_dim,
                                                                      hidden_decoder_dim, latent_dim, lr=learning_rate)

    cur_samples = None
    cur_elbo = None
    masked_batch_data = None

    # X_train_masked: array with 0s where the pixels are missing
    # and 1s where the pixels are not missing
    X_merged_masked = np.array(X_merged)
    X_merged_masked[np.where(X_merged_masked != missing_value)] = 1
    X_merged_masked[np.where(X_merged_masked == missing_value)] = 0

    non_zero_percentage = Utilities.non_zero_percentage(X_merged_masked)
    print('non missing values percentage: ' + str(non_zero_percentage) + ' %')

    X_filled = np.array(X_merged)

    print('')

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        iterations = int(N / batch_size)
        for i in range(iterations):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            batch_data = X_filled[start_index:end_index, :]

            cur_samples, cur_elbo = vae_in_pytorch.train(batch_data, batch_size, latent_dim, params, solver)

            masked_batch_data = X_merged_masked[start_index:end_index, :]
            cur_samples = np.multiply(masked_batch_data, batch_data) + np.multiply(1 - masked_batch_data, cur_samples)
            X_filled[start_index:end_index, :] = cur_samples

        print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))
    elapsed_time = time.time() - start_time

    print('training time: ' + str(elapsed_time))
    print('')

    X_filled[np.where(X_filled == missing_value)] = 1

    error1 = Utilities.rmse(X_merged, X_filled)
    print('root mean squared error: ' + str(error1))

    error2 = Utilities.mae(X_merged, X_filled)
    print('mean absolute error: ' + str(error2))

    X_filled_df = pd.DataFrame(X_filled)
    X_filled_df = X_filled_df.round(1)
    # X_filled_df = X_filled_df.replace(to_replace=0.0, value='---')
    X_filled_df.to_csv(path_or_buf=output_data_dir + '/users_movies_ratings_predicted_values.csv',
                       sep='\t', index=None, header=None)

    fig = plot_dataset_samples.plot_movielens_data(X_filled)
    fig.savefig(output_data_dir + '/average_movies_ratings.png', bbox_inches='tight')
    plt.close()

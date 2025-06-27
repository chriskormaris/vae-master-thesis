import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import *
from src.utilities import get_non_zero_percentage, rmse, mae
from src.utilities.get_movielens_dataset import get_movielens_dataset
from src.utilities.plot_utils import plot_movielens_data
from src.utilities.vae_in_pytorch import initialize_weights, train


def movielens(latent_dim=64, epochs=100, batch_size='250', learning_rate=0.01):
    missing_value = 0

    output_data_path = os.path.join(movielens_output_data_base_path, 'vaes_missing_values_in_pytorch')

    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)

    _, _, X_merged = get_movielens_dataset(movielens_dataset_path)

    num_users = X_merged.shape[0]
    num_movies = X_merged.shape[1]

    print(f'number of users: {num_users}')
    print(f'number of movies: {num_movies}')

    X_df = pd.DataFrame(X_merged)
    X_df = X_df.replace(to_replace=0.0, value='---')
    X_df.to_csv(
        path_or_buf=os.path.join(output_data_path, 'users_movies_ratings_missing_values.csv'),
        sep='\t',
        index=False,
        header=False
    )

    print()

    #####

    N = num_users
    input_dim = num_movies  # D
    # M1: number of neurons in the encoder
    # M2: number of neurons in the decoder
    hidden_encoder_dim = 400  # M1
    hidden_decoder_dim = hidden_encoder_dim  # M2
    # latent_dim = Z_dim
    if batch_size == 'N':
        batch_size = N
    else:
        batch_size = int(batch_size)

    #####

    params, solver = initialize_weights(
        input_dim,
        hidden_encoder_dim,
        hidden_decoder_dim,
        latent_dim,
        lr=learning_rate
    )

    cur_elbo = None

    # X_train_masked: array with 0s where the pixels are missing
    # and 1s where the pixels are not missing
    X_merged_masked = np.array(X_merged)
    X_merged_masked[np.where(X_merged_masked != missing_value)] = 1
    X_merged_masked[np.where(X_merged_masked == missing_value)] = 0

    non_zero_percentage = get_non_zero_percentage(X_merged_masked)
    print(f'non missing values percentage: {non_zero_percentage} %')

    X_filled = np.array(X_merged)

    print()

    iterations = int(N / batch_size)
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        for i in range(1, iterations + 1):
            start_index = (i - 1) * batch_size
            end_index = i * batch_size

            batch_data = X_filled[start_index:end_index, :]

            cur_samples, cur_elbo = train(batch_data, batch_size, latent_dim, params, solver)

            masked_batch_data = X_merged_masked[start_index:end_index, :]
            cur_samples = np.multiply(masked_batch_data, batch_data) + np.multiply(1 - masked_batch_data, cur_samples)
            X_filled[start_index:end_index, :] = cur_samples

        print(f'Epoch {epoch} | Loss (ELBO): {cur_elbo}')
    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print()

    X_filled[np.where(X_filled == missing_value)] = 1

    error1 = rmse(X_merged, X_filled)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_merged, X_filled)
    print(f'mean absolute error: {error2}')

    X_filled_df = pd.DataFrame(X_filled)
    X_filled_df = X_filled_df.round(1)
    # X_filled_df = X_filled_df.replace(to_replace=0.0, value='---')
    X_filled_df.to_csv(
        path_or_buf=os.path.join(output_data_path, 'users_movies_ratings_predicted_values.csv'),
        sep='\t',
        index=False,
        header=False
    )

    fig = plot_movielens_data(X_filled)
    fig.savefig(os.path.join(output_data_path, 'average_movies_ratings.png'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    movielens()

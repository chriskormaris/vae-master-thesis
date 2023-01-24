import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.utilities.constants import *
from src.utilities.get_movielens_dataset import get_movielens_dataset
from src.utilities.plot_utils import plot_movielens_data
from src.utilities.utils import get_non_zero_percentage, rmse, mae
from src.utilities.vae_in_tensorflow import vae


def movielens(latent_dim=64, epochs=100, batch_size='250', learning_rate=0.01):
    missing_value = 0

    output_data_path = movielens_output_data_base_path + 'vaes_missing_values_in_tensorflow/'
    logdir = tensorflow_logs_path + 'movielens_vae_missing_values'
    save_path = save_base_path + 'movielens_vae_missing_values'

    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    _, _, X_merged = get_movielens_dataset(movielens_dataset_path)

    num_users = X_merged.shape[0]
    num_movies = X_merged.shape[1]

    print(f'number of users: {num_users}')
    print(f'number of movies: {num_movies}')

    X_df = pd.DataFrame(X_merged)
    X_df = X_df.replace(to_replace=0.0, value='---')
    X_df.to_csv(
        path_or_buf=output_data_path + '/users_movies_ratings_missing_values.csv',
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

    x, loss_summ, apply_updates, summary_op, saver, elbo, x_recon_samples = vae(
        batch_size,
        input_dim,
        hidden_encoder_dim,
        hidden_decoder_dim,
        latent_dim,
        lr=learning_rate
    )

    cur_elbo = None

    # X_merged_masked: array with 0s where the pixels are missing
    # and 1s where the pixels are not missing
    X_merged_masked = np.array(X_merged)
    X_merged_masked[np.where(X_merged_masked != missing_value)] = 1
    X_merged_masked[np.where(X_merged_masked == missing_value)] = 0

    non_zero_percentage = get_non_zero_percentage(X_merged_masked)
    print(f'non missing values percentage: {non_zero_percentage} %')

    X_filled = np.array(X_merged)

    start_time = time.time()
    with tf.compat.v1.Session() as sess:
        summary_writer = tf.compat.v1.summary.FileWriter(logdir, graph=sess.graph)
        if os.path.isfile(save_path + '/model.ckpt'):
            print('Restoring saved parameters')
            saver.restore(sess, save_path + '/model.ckpt')
        else:
            print('Initializing parameters')
            sess.run(tf.compat.v1.global_variables_initializer())

        print()

        for epoch in range(1, epochs + 1):
            iterations = int(N / batch_size)
            for i in range(1, iterations + 1):
                start_index = (i - 1) * batch_size
                end_index = i * batch_size

                batch_data = X_filled[start_index:end_index, :]

                feed_dict = {x: batch_data}
                loss_str, _, summary_str, cur_elbo, cur_samples = sess.run(
                    [loss_summ, apply_updates, summary_op, elbo, x_recon_samples],
                    feed_dict=feed_dict
                )

                masked_batch_data = X_merged_masked[start_index:end_index, :]
                cur_samples = \
                    np.multiply(masked_batch_data, batch_data) + np.multiply(1 - masked_batch_data, cur_samples)
                X_filled[start_index:end_index, :] = cur_samples

                summary_writer.add_summary(loss_str, epoch)
                summary_writer.add_summary(summary_str, epoch)

            print(f'Epoch {epoch} | Loss (ELBO): {cur_elbo}')

            if epoch % 2 == 0:
                saver.save(sess, save_path + '/model.ckpt')
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
        path_or_buf=output_data_path + '/users_movies_ratings_predicted_values.csv',
        sep='\t',
        index=False,
        header=False
    )

    fig = plot_movielens_data(X_filled)
    fig.savefig(output_data_path + '/average_movies_ratings.png', bbox_inches='tight')
    plt.close()

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=./tensorflow_logs/movielens_vae_missing_values'.
    # Then open your browser and navigate to -> http://localhost:6006


if __name__ == '__main__':
    movielens()

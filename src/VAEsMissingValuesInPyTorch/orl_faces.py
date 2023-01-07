# THIS DATASET HAS VERY FEW EXAMPLES! #
# THE RESULTS ARE DISSATISFYING BECAUSE OF THAT. #

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src.Utilities.constants import *
from src.Utilities.get_orl_faces_dataset import get_orl_faces_dataset
from src.Utilities.plot_dataset_samples import plot_orl_faces
from src.Utilities.utils import reduce_data, construct_missing_data, get_non_zero_percentage, rmse, mae
from src.Utilities.vae_in_pytorch import initialize_weights, train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings


def orl_faces(latent_dim=64, epochs=100, batch_size='250', learning_rate=0.01, structured_or_random='structured'):
    missing_value = 0.5

    output_images_path = output_img_base_path + 'VAEsMissingValuesInPyTorch/orl_faces'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    # LOAD ORL FACES DATASET #
    X, y = get_orl_faces_dataset(orl_faces_dataset_path)

    # Reduce data to avoid memory error.
    X, y, _ = reduce_data(X, X.shape[0], 30000, y=y)

    #####

    X_missing, X, y = construct_missing_data(X, y, structured_or_random=structured_or_random)

    #####

    N = X.shape[0]
    input_dim = 10304  # D
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

    params, solver = initialize_weights(input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, lr=learning_rate)

    start_index = None
    end_index = None
    batch_labels = None
    cur_samples = None
    cur_elbo = None

    # X_train_masked: array with 0s where the pixels are missing
    # and 1s where the pixels are not missing
    X_train_masked = np.array(X_missing)
    X_train_masked[np.where(X_train_masked != missing_value)] = 1
    X_train_masked[np.where(X_train_masked == missing_value)] = 0

    non_zero_percentage = get_non_zero_percentage(X_train_masked)
    print('non missing values percentage: ' + str(non_zero_percentage) + ' %')

    X_filled = np.array(X_missing)

    print('')

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        iterations = int(N / batch_size)
        for i in range(iterations):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            batch_data = X_filled[start_index:end_index, :]
            batch_labels = y[start_index:end_index]

            cur_samples, cur_elbo = train(batch_data, batch_size, latent_dim, params, solver)

            masked_batch_data = X_train_masked[start_index:end_index, :]
            cur_samples = np.multiply(masked_batch_data, batch_data) + np.multiply(1 - masked_batch_data, cur_samples)
            X_filled[start_index:end_index, :] = cur_samples

        print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))

        if epoch == 1:
            for i in range(0, 40, 10):
                fig = plot_orl_faces(
                    X[start_index:end_index, :],
                    y[start_index:end_index],
                    categories=list(range(i, i + 10)),
                    title='Original Faces',
                    show_plot=False
                )
                fig.savefig(output_images_path + '/original_faces_' + str(i + 1) + '-' + str(i + 10) + '.png')
                plt.close()

            for i in range(0, 40, 10):
                fig = plot_orl_faces(
                    X_missing[start_index:end_index, :],
                    y[start_index:end_index],
                    categories=list(range(i, i + 10)),
                    title='Missing Faces',
                    show_plot=False
                )
                fig.savefig(output_images_path + '/missing_faces_' + str(i + 1) + '-' + str(i + 10) + '.png')
                plt.close()

            for i in range(0, 40, 10):
                fig = plot_orl_faces(
                    X_missing[start_index:end_index, :],
                    y[start_index:end_index],
                    categories=list(range(i, i + 10)),
                    title='Masked Faces',
                    show_plot=False
                )
                fig.savefig(output_images_path + '/masked_faces_' + str(i + 1) + '-' + str(i + 10) + '.png')
                plt.close()

        if epoch % 10 == 0 or epoch == 1:

            for i in range(0, 40, 10):
                fig = plot_orl_faces(
                    cur_samples,
                    batch_labels,
                    categories=list(range(i, i + 10)),
                    title='Epoch {}'.format(str(epoch).zfill(3)),
                    show_plot=False
                )
                fig.savefig(output_images_path + '/epoch_{}'
                            .format(str(epoch).zfill(3)) + '_faces_' + str(i + 1) + '-' + str(i + 10) + '.png')
                plt.close()
    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print('')

    error1 = rmse(X, X_filled)
    print(f'root mean squared error: {error1}')

    error2 = mae(X, X_filled)
    print(f'mean absolute error: {error2}')

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=./tensorflow_logs/faces'
    # Then open your browser ang navigate to -> http://localhost:6006


if __name__ == '__main__':
    orl_faces()

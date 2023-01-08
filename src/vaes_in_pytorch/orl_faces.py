# THIS DATASET HAS VERY FEW EXAMPLES! #

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src.utilities.constants import *
from src.utilities.get_orl_faces_dataset import get_orl_faces_dataset
from src.utilities.plot_dataset_samples import plot_orl_faces
from src.utilities.utils import rmse, mae
from src.utilities.vae_in_pytorch import initialize_weights, train


def orl_faces(latent_dim=64, epochs=100, batch_size='250', learning_rate=0.01):
    output_images_path = output_img_base_path + 'vaes_in_pytorch/orl_faces'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    # LOAD ORL FACES DATASET #
    X, y = get_orl_faces_dataset(orl_faces_dataset_path)

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

    for i in range(0, 40, 10):
        fig = plot_orl_faces(X, y, categories=list(range(i, i + 10)), title='Original Faces', show_plot=False)
        fig.savefig(output_images_path + '/original_faces_' + str(i + 1) + '-' + str(i + 10) + '.png')
        plt.close()

    #####

    params, solver = initialize_weights(input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, lr=learning_rate)

    batch_labels = None
    cur_samples = None
    cur_elbo = None
    X_recon = np.zeros((N, input_dim))

    print('')

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        iterations = int(N / batch_size)
        for i in range(iterations):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            batch_data = X[start_index:end_index, :]
            batch_labels = y[start_index:end_index]

            cur_samples, cur_elbo = train(batch_data, batch_size, latent_dim, params, solver)

            X_recon[start_index:end_index] = cur_samples

        print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))

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

    error1 = rmse(X, X_recon)
    print(f'root mean squared error: {error1}')

    error2 = mae(X, X_recon)
    print(f'mean absolute error: {error2}')

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=./tensorflow_logs/faces'
    # Then open your browser ang navigate to -> http://localhost:6006


if __name__ == '__main__':
    orl_faces()
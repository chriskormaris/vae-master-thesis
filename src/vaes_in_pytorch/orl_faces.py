# THIS DATASET HAS VERY FEW EXAMPLES! #

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src.utilities.constants import *
from src.utilities.get_orl_faces_dataset import get_orl_faces_dataset
from src.utilities.plot_utils import plot_images
from src.utilities.utils import rmse, mae
from src.utilities.vae_in_pytorch import initialize_weights, train


def orl_faces(latent_dim=64, epochs=100, batch_size='250', learning_rate=0.01):
    output_images_path = output_img_base_path + 'vaes_in_pytorch/orl_faces'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    # LOAD ORL FACES DATASET #
    X, y = get_orl_faces_dataset(orl_faces_dataset_path)

    #####

    for i in range(0, 40, 10):
        fig = plot_images(X, y, categories=list(range(i, i + 10)), title='Original Faces', show_plot=False)
        fig.savefig(f'{output_images_path}/original_faces_{i + 1}-{i + 10}.png')
        plt.close()

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

    cur_samples = None
    batch_labels = None
    cur_elbo = None
    X_recon = np.zeros((N, input_dim))

    print()

    iterations = int(N / batch_size)
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        for i in range(1, iterations + 1):
            start_index = (i - 1) * batch_size
            end_index = i * batch_size

            batch_data = X[start_index:end_index, :]
            batch_labels = y[start_index:end_index]

            cur_samples, cur_elbo = train(batch_data, batch_size, latent_dim, params, solver)

            X_recon[start_index:end_index] = cur_samples

        print(f'Epoch {epoch} | Loss (ELBO): {cur_elbo}')

        if epoch % 10 == 0 or epoch == 1:
            if cur_samples is not None:
                fig = plot_images(
                    cur_samples,
                    batch_labels,
                    categories=list(range(10)),
                    title=f'Epoch {str(epoch).zfill(3)}',
                    show_plot=False
                )
                fig.savefig(f'{output_images_path}/epoch_{str(epoch).zfill(3)}_faces_1-10.png')
                plt.close()

    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print()

    error1 = rmse(X, X_recon)
    print(f'root mean squared error: {error1}')

    error2 = mae(X, X_recon)
    print(f'mean absolute error: {error2}')

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=./tensorflow_logs/faces'
    # Then open your browser ang navigate to -> http://localhost:6006


if __name__ == '__main__':
    orl_faces()

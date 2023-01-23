import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src.utilities.constants import *
from src.utilities.get_binarized_mnist_dataset import get_binarized_mnist_dataset, get_binarized_mnist_labels
from src.utilities.get_binarized_mnist_dataset import obtain
from src.utilities.plot_dataset_samples import plot_mnist_or_omniglot_data
from src.utilities.utils import rmse, mae
from src.utilities.vae_in_pytorch import initialize_weights, train


def binarized_mnist(latent_dim=64, epochs=100, batch_size='250', learning_rate=0.01):
    output_images_path = output_img_base_path + 'vaes_in_pytorch/binarized_mnist'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    if not os.path.exists(binarized_dataset_path):
        os.makedirs(binarized_dataset_path)
        obtain(binarized_dataset_path)

    X_test = get_binarized_mnist_dataset(binarized_dataset_path + 'binarized_mnist_test.amat', 'TEST')
    y_test = get_binarized_mnist_labels(binarized_dataset_path + 'binarized_mnist_test_labels.txt', 'TEST')

    # random shuffle the data
    np.random.seed(0)
    s = np.random.permutation(X_test.shape[0])
    X_test = X_test[s, :]
    y_test = y_test[s]

    print()

    #####

    N = X_test.shape[0]
    input_dim = 784  # D
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

    fig = plot_mnist_or_omniglot_data(X_test, y_test, title='Original Data')
    fig.savefig(f'{output_images_path}/original_data.png', bbox_inches='tight')
    plt.close()

    #####

    params, solver = initialize_weights(input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, lr=learning_rate)

    cur_samples = None
    batch_labels = None
    cur_elbo = None
    X_recon = np.zeros((N, input_dim))

    iterations = int(N / batch_size)
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        for i in range(1, iterations + 1):
            start_index = (i - 1) * batch_size
            end_index = i * batch_size

            batch_data = X_test[start_index:end_index, :]
            batch_labels = y_test[start_index:end_index]

            cur_samples, cur_elbo = train(batch_data, batch_size, latent_dim, params, solver)

            X_recon[start_index:end_index, :] = cur_samples

        print(f'Epoch {epoch} | Loss (ELBO): {cur_elbo}')

        if epoch % 10 == 0 or epoch == 1:
            fig = plot_mnist_or_omniglot_data(cur_samples, batch_labels, title=f'Epoch {str(epoch).zfill(3)}')
            fig.savefig(f'{output_images_path}/epoch_{str(epoch).zfill(3)}.png', bbox_inches='tight')
            plt.close()
    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print()

    error1 = rmse(X_test, X_recon)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_test, X_recon)
    print(f'mean absolute error: {error2}')


if __name__ == '__main__':
    binarized_mnist()

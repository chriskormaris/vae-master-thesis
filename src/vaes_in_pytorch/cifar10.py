# WORKS WELL ONLY WITH GRAYSCALED IMAGES! RGB IMAGES HAVE DISSATISFYING RESULTS. #

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10 as cifar10_dataset

from src.utilities.constants import *
from src.utilities.plot_dataset_samples import plot_images
from src.utilities.utils import rmse, mae
from src.utilities.vae_in_pytorch import initialize_weights, train


def cifar10(latent_dim=64, epochs=100, batch_size='250', learning_rate=0.01, rgb_or_grayscale='grayscale', category=3):
    if rgb_or_grayscale.lower() == 'grayscale':
        output_images_path = output_img_base_path + 'vaes_in_pytorch/cifar10_grayscale'
    else:
        output_images_path = output_img_base_path + 'vaes_in_pytorch/cifar10_rgb'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    # LOAD CIFAR-10 DATASET #
    (X_train, y_train), (X_test, y_test) = cifar10_dataset.load_data()

    X_train = X_train[np.where(y_train == category)[0], :]
    y_train = y_train[np.where(y_train == category)[0]]
    X_test = X_test[np.where(y_test == category)[0], :]
    y_test = y_test[np.where(y_test == category)[0]]

    # Merge train and test data together to increase the train dataset size.
    X_train = np.concatenate((X_train, X_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)

    # randomize data
    s = np.random.permutation(X_train.shape[0])
    X_train = X_train[s, :]
    y_train = y_train[s]

    if rgb_or_grayscale.lower() == 'grayscale':
        # Convert colored images from 3072 dimensions to 1024 grayscale images.
        X_train = np.dot(X_train[:, :, :, :3], [0.299, 0.587, 0.114])
        X_train = np.reshape(X_train, newshape=(-1, 1024))  # X_merged: N x 1024
    else:
        # We will flatten the 32x32 images into vectors of size 3072.
        X_train = X_train.reshape((-1, 3072))

    # We will normalize all values between 0 and 1.
    X_train = X_train / 255.

    #####

    N = X_train.shape[0]
    input_dim = X_train.shape[1]  # D: 3072 or 1024
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

    fig = None
    start_index = None
    end_index = None
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

            batch_data = X_train[start_index:end_index, :]
            batch_labels = y_train[start_index:end_index]

            cur_samples, cur_elbo = train(batch_data, batch_size, latent_dim, params, solver)

            X_recon[start_index:end_index, :] = cur_samples

        print(f'Epoch {epoch} | Loss (ELBO): {cur_elbo}')

        if epoch == 1:
            fig = plot_images(
                X=X_train[start_index:end_index, :],
                y=batch_labels,
                categories=[category],
                n=100,
                grayscale=True if input_dim == 1024 else False
            )
            fig.savefig(f'{output_images_path}/original_data.png', bbox_inches='tight')
            plt.close()

        if epoch % 10 == 0 or epoch == 1:
            fig = plot_images(
                X=cur_samples,
                y=batch_labels,
                categories=[category],
                n=100,
                title=f'Epoch {str(epoch).zfill(3)}',
                grayscale=True if input_dim == 1024 else False
            )
            fig.savefig(f'{output_images_path}/epoch_{str(epoch).zfill(3)}.png', bbox_inches='tight')
            plt.close()

    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print()

    error1 = rmse(X_train, X_recon)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_train, X_recon)
    print(f'mean absolute error: {error2}')


if __name__ == '__main__':
    cifar10()

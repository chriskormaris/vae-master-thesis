# WORKS WELL ONLY WITH GRAYSCALED IMAGES! RGB IMAGES HAVE DISSATISFYING RESULTS. #

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src.utilities.constants import *
from src.utilities.get_cifar10_dataset import get_cifar10_dataset
from src.utilities.plot_dataset_samples import plot_cifar10_data
from src.utilities.utils import rmse, mae
from src.utilities.vae_in_pytorch import initialize_weights, train


def cifar10(latent_dim=64, epochs=100, batch_size='250', learning_rate=0.01, rgb_or_grayscale='grayscale'):
    if rgb_or_grayscale.lower() == 'grayscale':
        output_images_path = output_img_base_path + 'vaes_in_pytorch/cifar10_grayscale'
    else:
        output_images_path = output_img_base_path + 'vaes_in_pytorch/cifar10_rgb'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    # LOAD CIFAR-10 DATASET #
    (X_train, y_train), (X_test, y_test) = get_cifar10_dataset(cifar10_dataset_path)

    N = X_train.shape[0]

    # We will normalize all values between 0 and 1,
    # and we will flatten the 32x32 images into vectors of size 3072.

    # Reduce train and test data to only two categories, the class 3 ('cat') and the class 5 ('dog').
    categories = [3, 5]

    category = 3
    X_train_cats = X_train[np.where(y_train == category)[0], :]
    y_train_cats = y_train[np.where(y_train == category)[0]]
    X_test_cats = X_test[np.where(y_test == category)[0], :]
    y_test_cats = y_test[np.where(y_test == category)[0]]

    category = 5
    X_train_dogs = X_train[np.where(y_train == category)[0], :]
    y_train_dogs = y_train[np.where(y_train == category)[0]]
    X_test_dogs = X_test[np.where(y_test == category)[0], :]
    y_test_dogs = y_test[np.where(y_test == category)[0]]

    X_train = np.concatenate((X_train_cats, X_train_dogs), axis=0)
    y_train = np.concatenate((y_train_cats, y_train_dogs), axis=0)
    X_test = np.concatenate((X_test_cats, X_test_dogs), axis=0)
    y_test = np.concatenate((y_test_cats, y_test_dogs), axis=0)

    # Merge train and test data together to increase the train dataset size.
    X_train = np.concatenate((X_train, X_test), axis=0)  # X_train: N x 3072
    y_train = np.concatenate((y_train, y_test), axis=0)

    # randomize data
    s = np.random.permutation(X_train.shape[0])
    X_train = X_train[s, :]
    y_train = y_train[s]

    if rgb_or_grayscale.lower() == 'grayscale':
        # Convert colored images from 3072 dimensions to 1024 grayscale images.
        X_train = np.dot(X_train[:, :, :, :3], [0.299, 0.587, 0.114])
        X_train = np.reshape(X_train, newshape=(-1, 1024))  # X_train: N x 1024
    else:
        # We will normalize all values between 0 and 1,
        # and we will flatten the 32x32 images into vectors of size 3072.
        X_train = X_train.reshape((len(X_train), 3072))

    X_train = X_train.astype('float32') / 255.

    #####

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
    batch_labels = None
    cur_samples = None
    cur_elbo = None
    X_recon = np.zeros((N, input_dim))

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        iterations = int(N / batch_size)
        for i in range(iterations):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            batch_data = X_train[start_index:end_index, :]
            batch_labels = y_train[start_index:end_index]

            cur_samples, cur_elbo = train(batch_data, batch_size, latent_dim, params, solver)

            X_recon[start_index:end_index, :] = cur_samples

        print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))

        if epoch == 1:
            if input_dim == 1024:
                fig = plot_cifar10_data(X_train[start_index:end_index, :], y_train[start_index:end_index],
                                        categories=categories, grayscale=True)
            elif input_dim == 3072:
                fig = plot_cifar10_data(X_train[start_index:end_index, :], y_train[start_index:end_index],
                                        categories=categories, grayscale=False)
            fig.savefig(output_images_path + '/original_data.png', bbox_inches='tight')
            plt.close()

        if epoch % 10 == 0 or epoch == 1:

            if input_dim == 1024:
                fig = plot_cifar10_data(cur_samples, batch_labels, title='Epoch {}'.format(str(epoch).zfill(3)),
                                        categories=categories, grayscale=True)
            elif input_dim == 3072:
                fig = plot_cifar10_data(cur_samples, batch_labels, title='Epoch {}'.format(str(epoch).zfill(3)),
                                        categories=categories, grayscale=False)
            fig.savefig(output_images_path + '/epoch_{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print('')

    error1 = rmse(X_train, X_recon)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_train, X_recon)
    print(f'mean absolute error: {error2}')


if __name__ == '__main__':
    cifar10()

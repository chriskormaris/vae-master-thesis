import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist as fashion_mnist_dataset
from keras.datasets import mnist as mnist_dataset

from src.utilities.constants import *
from src.utilities.plot_dataset_samples import plot_mnist_or_omniglot_data
from src.utilities.utils import reduce_data, construct_missing_data, get_non_zero_percentage, rmse, mae
from src.utilities.vae_in_pytorch import initialize_weights, train


def mnist(
        latent_dim=64,
        epochs=100,
        batch_size='N',
        learning_rate=0.01,
        structured_or_random='structured',
        digits_or_fashion='digits'
):
    missing_value = 0.5

    if digits_or_fashion == 'digits':
        output_images_path = output_img_base_path + 'vaes_missing_values_in_pytorch/mnist'
        mnist_data = mnist_dataset.load_data(os.getcwd() + '\\' + mnist_dataset_path + 'mnist.npz')
    else:
        output_images_path = output_img_base_path + 'vaes_missing_values_in_pytorch/fashion_mnist'
        mnist_data = fashion_mnist_dataset.load_data()

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    X_train, y_train = mnist_data[0]

    # We will normalize all values between 0 and 1,
    # and we will flatten the 28x28 images into vectors of size 784.
    X_train = X_train / 255.
    X_train = X_train.reshape((-1, np.prod(X_train.shape[1:])))

    #####

    # Reduce data to avoid memory error.
    X_train, y_train, _ = reduce_data(X_train, X_train.shape[0], 30000, y=y_train)

    # construct data with missing values
    X_train_missing, X_train, y_train = construct_missing_data(
        X_train,
        y_train,
        structured_or_random=structured_or_random
    )

    print('')

    #####

    N = X_train.shape[0]
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

    params, solver = initialize_weights(
        input_dim,
        hidden_encoder_dim,
        hidden_decoder_dim,
        latent_dim,
        lr=learning_rate
    )

    start_index = None
    end_index = None
    batch_labels = None
    cur_samples = None
    cur_elbo = None
    masked_batch_data = None

    # X_train_masked: array with 0s where the pixels are missing
    # and 1s where the pixels are not missing
    X_train_masked = np.array(X_train_missing)
    X_train_masked[np.where(X_train_masked != missing_value)] = 1
    X_train_masked[np.where(X_train_masked == missing_value)] = 0

    non_zero_percentage = get_non_zero_percentage(X_train_masked)
    print('non missing values percentage: ' + str(non_zero_percentage) + ' %')

    X_filled = np.array(X_train_missing)

    print('')

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        iterations = int(N / batch_size)
        for i in range(iterations):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            batch_data = X_filled[start_index:end_index, :]
            batch_labels = y_train[start_index:end_index]

            cur_samples, cur_elbo = train(batch_data, batch_size, latent_dim, params, solver)

            masked_batch_data = X_train_masked[start_index:end_index, :]
            cur_samples = np.multiply(masked_batch_data, batch_data) + np.multiply(1 - masked_batch_data, cur_samples)
            X_filled[start_index:end_index, :] = cur_samples

        print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))

        if epoch == 1:
            fig = plot_mnist_or_omniglot_data(
                X_train[start_index:end_index, :],
                batch_labels,
                title='Original Data'
            )
            fig.savefig(output_images_path + '/original_data.png', bbox_inches='tight')
            plt.close()

            fig = plot_mnist_or_omniglot_data(
                X_train_missing[start_index:end_index, :],
                batch_labels,
                title='Original Data'
            )
            fig.savefig(output_images_path + '/missing_data.png', bbox_inches='tight')
            plt.close()

            fig = plot_mnist_or_omniglot_data(masked_batch_data, batch_labels, title='Masked Data')
            fig.savefig(output_images_path + '/masked_data.png', bbox_inches='tight')
            plt.close()

        if epoch % 10 == 0 or epoch == 1:
            fig = plot_mnist_or_omniglot_data(
                cur_samples,
                batch_labels,
                title='Epoch {}'.format(str(epoch).zfill(3))
            )
            fig.savefig(output_images_path + '/epoch_{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print('')

    error1 = rmse(X_train, X_filled)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_train, X_filled)
    print(f'mean absolute error: {error2}')


if __name__ == '__main__':
    mnist()

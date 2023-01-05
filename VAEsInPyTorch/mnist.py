import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets.mnist import load_data

from Utilities.Utilities import rmse, mae
from Utilities.initialize_weights_in_pytorch import initialize_weights
from Utilities.plot_dataset_samples import plot_mnist_or_omniglot_data
from Utilities.vae_in_pytorch import train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings


def mnist(latent_dim=64, epochs=100, batch_size=250, learning_rate=0.01, digits_or_fashion='digits'):
    if digits_or_fashion == 'digits':
        output_images_dir = './output_images/VAEsInPyTorch/mnist'
        mnist_dataset_dir = '../MNIST_dataset'
    else:
        output_images_dir = './output_images/VAEsInPyTorch/fashion_mnist'
        mnist_dataset_dir = '../FASHION_MNIST_dataset'

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    mnist = load_data(mnist_dataset_dir)

    X_train = mnist[0][0]
    X_train = X_train.reshape(-1, 784)
    y_train = mnist[0][1]

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

    fig = plot_mnist_or_omniglot_data(X_train, y_train, title='Original Data')
    fig.savefig(output_images_dir + '/original_data.png', bbox_inches='tight')
    plt.close()

    #####

    params, solver = initialize_weights(input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, lr=learning_rate)

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

            # batch_data, batch_labels = mnist.train.next_batch(batch_size=batch_size)
            # ALTERNATIVE
            batch_data = X_train[start_index:end_index, :]
            batch_labels = y_train[start_index:end_index]

            cur_samples, cur_elbo = train(batch_data, batch_size, latent_dim, params, solver)

            X_recon[start_index:end_index, :] = cur_samples

        print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))

        if epoch % 10 == 0 or epoch == 1:
            fig = plot_mnist_or_omniglot_data(cur_samples, batch_labels, title='Epoch {}'.format(str(epoch).zfill(3)))
            fig.savefig(output_images_dir + '/epoch_{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
    elapsed_time = time.time() - start_time

    print('training time: ' + str(elapsed_time))
    print('')

    error1 = rmse(X_train, X_recon)
    print('root mean squared error: ' + str(error1))

    error2 = mae(X_train, X_recon)
    print('mean absolute error: ' + str(error2))

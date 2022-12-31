import inspect
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from __init__ import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings

__author__ = 'c.kormaris'


#####

# MAIN #

if __name__ == '__main__':

    output_images_dir = './output_images/VAEsInPyTorch/binarized_mnist'
    binarized_mnist_dataset_dir = '../Binarized_MNIST_dataset'

    if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)

    if not os.path.exists(binarized_mnist_dataset_dir):
        os.makedirs(binarized_mnist_dataset_dir)
        get_binarized_mnist_dataset.obtain(binarized_mnist_dataset_dir)

    X_test = get_binarized_mnist_dataset.get_binarized_mnist_dataset(binarized_mnist_dataset_dir
                                                                     + '/binarized_mnist_test.amat', 'TEST')
    y_test = get_binarized_mnist_dataset.get_binarized_mnist_labels(binarized_mnist_dataset_dir
                                                                    + '/binarized_mnist_test_labels.txt', 'TEST')

    # random shuffle the data
    np.random.seed(0)
    s = np.random.permutation(X_test.shape[0])
    X_test = X_test[s, :]
    y_test = y_test[s]

    print('')

    #####

    N = X_test.shape[0]
    input_dim = 784  # D
    # M1: number of neurons in the encoder
    # M2: number of neurons in the decoder
    hidden_encoder_dim = 400  # M1
    hidden_decoder_dim = hidden_encoder_dim  # M2
    latent_dim = 64  # Z_dim
    epochs = 50
    batch_size = 250
    learning_rate = 0.01

    #####

    fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_test, y_test, title='Original Data')
    fig.savefig(output_images_dir + '/original_data.png', bbox_inches='tight')
    plt.close()

    #####

    params, solver = vae_in_pytorch.initialize_weights(input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, lr=learning_rate)

    cur_samples = None
    batch_labels = None
    cur_elbo = None
    X_recon = np.zeros((N, input_dim))

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        iterations = int(N / batch_size)
        for i in range(iterations):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            batch_data = X_test[start_index:end_index, :]
            batch_labels = y_test[start_index:end_index]

            cur_samples, cur_elbo = vae_in_pytorch.train(batch_data, batch_size, latent_dim, params, solver)

            X_recon[start_index:end_index, :] = cur_samples

        print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))

        if epoch % 10 == 0 or epoch == 1:

            fig = plot_dataset_samples.plot_mnist_or_omniglot_data(cur_samples, batch_labels, title='Epoch {}'.format(str(epoch).zfill(3)))
            fig.savefig(output_images_dir + '/epoch_{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
    elapsed_time = time.time() - start_time

    print('training time: ' + str(elapsed_time))
    print('')

    error1 = Utilities.rmse(X_test, X_recon)
    print('root mean squared error: ' + str(error1))

    error2 = Utilities.mae(X_test, X_recon)
    print('mean absolute error: ' + str(error2))

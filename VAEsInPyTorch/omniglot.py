# WORKS BETTER WITH THE ENGLISH ALPHABET! THE GREEK ALPHABET HAS VERY FEW EXAMPLES. #

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


#####

# MAIN #

if __name__ == '__main__':

    omniglot_dataset_dir = '../OMNIGLOT_dataset'

    language = sys.argv[5]

    if language.lower() == 'greek':
        output_images_dir = './output_images/VAEsInPyTorch/omniglot_greek'
        alphabet = 20
    else:
        output_images_dir = './output_images/VAEsInPyTorch/omniglot_english'
        alphabet = 31

    if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)

    # LOAD OMNIGLOT DATASET #
    X_train, y_train = get_omniglot_dataset.get_omniglot_dataset(omniglot_dataset_dir + '/chardata.mat',
                                                                 TrainOrTest='train', alphabet=alphabet, binarize=True)
    X_test, y_test = get_omniglot_dataset.get_omniglot_dataset(omniglot_dataset_dir + '/chardata.mat',
                                                               TrainOrTest='test', alphabet=alphabet, binarize=True)

    X_merged = np.concatenate((X_train, X_test), axis=0)
    y_merged = np.concatenate((y_train, y_test), axis=0)

    #####

    N = X_merged.shape[0]
    input_dim = 784  # D
    # M1: number of neurons in the encoder
    # M2: number of neurons in the decoder
    hidden_encoder_dim = 400  # M1
    hidden_decoder_dim = hidden_encoder_dim  # M2
    latent_dim = int(sys.argv[1])  # Z_dim
    epochs = int(sys.argv[2])
    batch_size = sys.argv[3]
    if batch_size == 'N':
        batch_size = N
    else:
        batch_size = int(batch_size)
    learning_rate = float(sys.argv[4])

    #####

    fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged, y_merged, categories=list(range(1, 11)), title='Original Data')
    fig.savefig(output_images_dir + '/original_data_characters_1-10.png', bbox_inches='tight')
    plt.close()
    fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged, y_merged, categories=list(range(11, 21)), title='Original Data')
    fig.savefig(output_images_dir + '/original_data_characters_11-20.png', bbox_inches='tight')
    plt.close()
    if language.lower() == 'greek':
        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged, y_merged, categories=list(range(21, 25)), title='Original Data')
        fig.savefig(output_images_dir + '/original_data_characters_21-24.png', bbox_inches='tight')
    else:
        fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged, y_merged, categories=list(range(21, 27)), title='Original Data')
        fig.savefig(output_images_dir + '/original_data_characters_21-26.png', bbox_inches='tight')
    plt.close()

    #####

    params, solver = initialize_weights_in_pytorch.initialize_weights(input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, lr=learning_rate)

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

            batch_data = X_merged[start_index:end_index, :]
            batch_labels = y_merged[start_index:end_index]

            cur_samples, cur_elbo = vae_in_pytorch.train(batch_data, batch_size, latent_dim, params, solver)

            X_recon[start_index:end_index] = cur_samples

        print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))

        if epoch % 10 == 0 or epoch == 1:

            fig = plot_dataset_samples.plot_mnist_or_omniglot_data(cur_samples, batch_labels, categories=list(range(1, 11)), n=5, title='Epoch {}'.format(str(epoch).zfill(3)))
            fig.savefig(output_images_dir + '/epoch_{}_characters_1-10.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
            fig = plot_dataset_samples.plot_mnist_or_omniglot_data(cur_samples, batch_labels, categories=list(range(11, 21)), n=5, title='Epoch {}'.format(str(epoch).zfill(3)))
            fig.savefig(output_images_dir + '/epoch_{}_characters_11-20.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
            if language.lower() == 'greek':
                fig = plot_dataset_samples.plot_mnist_or_omniglot_data(cur_samples, batch_labels, categories=list(range(21, 25)), n=5, title='Epoch {}'.format(str(epoch).zfill(3)))
                fig.savefig(output_images_dir + '/epoch_{}_characters_21-24.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            else:
                fig = plot_dataset_samples.plot_mnist_or_omniglot_data(cur_samples, batch_labels, categories=list(range(21, 27)), n=5, title='Epoch {}'.format(str(epoch).zfill(3)))
                fig.savefig(output_images_dir + '/epoch_{}_characters_21-26.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
    elapsed_time = time.time() - start_time

    print('training time: ' + str(elapsed_time))
    print('')

    error1 = Utilities.rmse(X_merged, X_recon)
    print('root mean squared error: ' + str(error1))

    error2 = Utilities.mae(X_merged, X_recon)
    print('mean absolute error: ' + str(error2))

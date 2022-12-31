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

missing_value = 0.5

#####

# MAIN #

if __name__ == '__main__':

    omniglot_dataset_dir = '../OMNIGLOT_dataset'

    language = sys.argv[6]

    if language.lower() == 'greek':
        output_images_dir = './output_images/VAEsMissingValuesInPyTorch/omniglot_greek'
        alphabet = 20
    else:
        output_images_dir = './output_images/VAEsMissingValuesInPyTorch/omniglot_english'
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

    X_merged_missing, X_merged, y_merged = Utilities.construct_missing_data(X_merged, y_merged, structured_or_random=sys.argv[5])

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

    params, solver = initialize_weights_in_pytorch.initialize_weights(input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, lr=learning_rate)

    start_index = None
    end_index = None
    batch_labels = None
    cur_samples = None
    cur_elbo = None
    masked_batch_data = None

    # X_merged_masked: array with 0s where the pixels are missing
    # and 1s where the pixels are not missing
    X_merged_masked = np.array(X_merged_missing)
    X_merged_masked[np.where(X_merged_masked != missing_value)] = 1
    X_merged_masked[np.where(X_merged_masked == missing_value)] = 0

    non_zero_percentage = Utilities.non_zero_percentage(X_merged_masked)
    print('non missing values percentage: ' + str(non_zero_percentage) + ' %')

    X_filled = np.array(X_merged_missing)

    print('')

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        iterations = int(N / batch_size)
        for i in range(iterations):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            batch_data = X_filled[start_index:end_index, :]
            batch_labels = y_merged[start_index:end_index]

            cur_samples, cur_elbo = vae_in_pytorch.train(batch_data, batch_size, latent_dim, params, solver)

            masked_batch_data = X_merged_masked[start_index:end_index, :]
            cur_samples = np.multiply(masked_batch_data, batch_data) + np.multiply(1 - masked_batch_data, cur_samples)
            X_filled[start_index:end_index, :] = cur_samples

        print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))

        if epoch == 1:

            fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged[start_index:end_index, :], y_merged[start_index:end_index], categories=list(range(1, 11)), title='Original Data')
            fig.savefig(output_images_dir + '/original_data_characters_1-10.png', bbox_inches='tight')
            plt.close()
            fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged[start_index:end_index, :], y_merged[start_index:end_index], categories=list(range(11, 21)), title='Original Data')
            fig.savefig(output_images_dir + '/original_data_characters_11-20.png', bbox_inches='tight')
            plt.close()
            if language.lower() == 'greek':
                fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged[start_index:end_index, :], y_merged[start_index:end_index], categories=list(range(21, 24)), title='Original Data')
                fig.savefig(output_images_dir + '/original_data_characters_21-24.png', bbox_inches='tight')
            else:
                fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged[start_index:end_index, :], y_merged[start_index:end_index], categories=list(range(21, 27)), title='Original Data')
                fig.savefig(output_images_dir + '/original_data_characters_21-26.png', bbox_inches='tight')
            plt.close()

            fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged_missing[start_index:end_index, :], y_merged[start_index:end_index], categories=list(range(1, 11)), title='Original Data')
            fig.savefig(output_images_dir + '/missing_data_characters_1-10.png', bbox_inches='tight')
            plt.close()
            fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged_missing[start_index:end_index, :], y_merged[start_index:end_index], categories=list(range(11, 21)), title='Original Data')
            fig.savefig(output_images_dir + '/missing_data_characters_11-20.png', bbox_inches='tight')
            plt.close()
            if language.lower() == 'greek':
                fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged_missing[start_index:end_index, :], y_merged[start_index:end_index], categories=list(range(21, 25)), title='Original Data')
                fig.savefig(output_images_dir + '/missing_data_characters_21-24.png', bbox_inches='tight')
            else:
                fig = plot_dataset_samples.plot_mnist_or_omniglot_data(X_merged_missing[start_index:end_index, :], y_merged[start_index:end_index], categories=list(range(21, 27)), title='Epoch {}'.format(str(epoch).zfill(3)))
                fig.savefig(output_images_dir + '/missing_data_characters_21-26.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()

            fig = plot_dataset_samples.plot_mnist_or_omniglot_data(masked_batch_data, batch_labels, categories=list(range(1, 11)), title='Masked Data')
            fig.savefig(output_images_dir + '/masked_data_characters_1-10.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
            fig = plot_dataset_samples.plot_mnist_or_omniglot_data(masked_batch_data, batch_labels, categories=list(range(1, 11)), title='Masked Data')
            fig.savefig(output_images_dir + '/masked_data_characters_11-20.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
            if language.lower() == 'greek':
                fig = plot_dataset_samples.plot_mnist_or_omniglot_data(masked_batch_data, batch_labels, categories=list(range(1, 11)), title='Masked Data')
                fig.savefig(output_images_dir + '/masked_data_characters_21-24.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            else:
                fig = plot_dataset_samples.plot_mnist_or_omniglot_data(masked_batch_data, batch_labels, categories=list(range(1, 11)), title='Masked Data')
                fig.savefig(output_images_dir + '/masked_data_characters_21-26.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()

        if epoch % 10 == 0 or epoch == 1:

            fig = plot_dataset_samples.plot_mnist_or_omniglot_data(cur_samples, batch_labels, categories=list(range(1, 11)), title='Epoch {}'.format(str(epoch).zfill(3)))
            fig.savefig(output_images_dir + '/epoch_{}_characters_1-10.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
            fig = plot_dataset_samples.plot_mnist_or_omniglot_data(cur_samples, batch_labels, categories=list(range(11, 21)), title='Epoch {}'.format(str(epoch).zfill(3)))
            fig.savefig(output_images_dir + '/epoch_{}_characters_11-20.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
            if language.lower() == 'greek':
                fig = plot_dataset_samples.plot_mnist_or_omniglot_data(cur_samples, batch_labels, categories=list(range(21, 25)), title='Epoch {}'.format(str(epoch).zfill(3)))
                fig.savefig(output_images_dir + '/epoch_{}_characters_21-24.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            else:
                fig = plot_dataset_samples.plot_mnist_or_omniglot_data(cur_samples, batch_labels, categories=list(range(21, 27)), title='Epoch {}'.format(str(epoch).zfill(3)))
                fig.savefig(output_images_dir + '/epoch_{}_characters_21-26.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
    elapsed_time = time.time() - start_time

    print('training time: ' + str(elapsed_time))
    print('')

    error1 = Utilities.rmse(X_merged, X_filled)
    print('root mean squared error: ' + str(error1))

    error2 = Utilities.mae(X_merged, X_filled)
    print('mean absolute error: ' + str(error2))

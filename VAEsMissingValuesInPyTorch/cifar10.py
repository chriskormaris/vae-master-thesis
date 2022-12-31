# WORKS WELL ONLY WITH GRAYSCALED IMAGES! RGB IMAGES HAVE DISSATISFYING RESULTS. #
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

    RGBOrGrayscale = sys.argv[6]

    if RGBOrGrayscale.lower() == 'grayscale':
        output_images_dir = './output_images/VAEsMissingValuesInPyTorch/cifar10_grayscale'
    else:
        output_images_dir = './output_images/VAEsMissingValuesInPyTorch/cifar10_rgb'

    cifar_10_dataset_dir = '../CIFAR_dataset/CIFAR-10'

    if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)

    # LOAD CIFAR-10 DATASET #
    (X_train, y_train), (X_test, y_test) = get_cifar10_dataset.get_cifar10_dataset(cifar_10_dataset_dir)

    # Reduce data to avoid memory error.
    X_train, y_train, _ = Utilities.reduce_data(X_train, X_train.shape[0], 15000, y=y_train)
    X_test, y_test, _ = Utilities.reduce_data(X_test, X_test.shape[0], 15000, y=y_test)

    # reduce train and test data to only two categories, the class 3 ('cat') and the class 5 ('dog')
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

    # merge train and test data together to increase the train dataset size
    X_train = np.concatenate((X_train, X_test), axis=0)  # X_train: N x 3072
    y_train = np.concatenate((y_train, y_test), axis=0)

    # randomize data
    s = np.random.permutation(X_train.shape[0])
    X_train = X_train[s, :]
    y_train = y_train[s]

    if RGBOrGrayscale.lower() == 'grayscale':
        # convert colored images from 3072 dimensions to 1024 grayscale images
        X_train = np.dot(X_train[:, :, :, :3], [0.299, 0.587, 0.114])
        X_train = np.reshape(X_train, newshape=(-1, 1024))  # X_train: N x 1024
        X_test = np.dot(X_test[:, :, :, :3], [0.299, 0.587, 0.114])
        X_test = np.reshape(X_test, newshape=(-1, 1024))  # X_test: N x 1024
    else:
        # We will normalize all values between 0 and 1
        # and we will flatten the 32x32 images into vectors of size 3072.
        X_train = X_train.reshape((len(X_train), 3072))
        X_test = X_test.reshape((len(X_test), 3072))

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
	
    #####

    # construct data with missing values
    X_train_missing, X_train, y_train = Utilities.construct_missing_data(X_train, y_train, structured_or_random=sys.argv[5])

    #####

    N = X_train.shape[0]
    input_dim = X_train.shape[1]  # D: 3072 or 1024
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

    fig = None
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

    non_zero_percentage = Utilities.non_zero_percentage(X_train_masked)
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

            cur_samples, cur_elbo = vae_in_pytorch.train(batch_data, batch_size, latent_dim, params, solver)

            masked_batch_data = X_train_masked[start_index:end_index, :]
            cur_samples = np.multiply(masked_batch_data, batch_data) + np.multiply(1 - masked_batch_data, cur_samples)
            X_filled[start_index:end_index, :] = cur_samples

        print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))

        if epoch == 1:
            if input_dim == 1024:
                fig = plot_dataset_samples.plot_cifar10_data(X_train[start_index:end_index, :], y_train[start_index:end_index],
                                                             categories=categories, grayscale=True)
            elif input_dim == 3072:
                fig = plot_dataset_samples.plot_cifar10_data(X_train[start_index:end_index, :], y_train[start_index:end_index],
                                                             categories=categories, grayscale=False)
            fig.savefig(output_images_dir + '/original_data.png', bbox_inches='tight')
            plt.close()

            if input_dim == 1024:
                fig = plot_dataset_samples.plot_cifar10_data(X_train_missing[start_index:end_index, :], y_train[start_index:end_index],
                                                             categories=categories, grayscale=True)
            elif input_dim == 3072:
                fig = plot_dataset_samples.plot_cifar10_data(X_train_missing[start_index:end_index, :], y_train[start_index:end_index],
                                                             categories=categories, grayscale=False)
            fig.savefig(output_images_dir + '/missing_data.png', bbox_inches='tight')
            plt.close()

        if epoch % 10 == 0 or epoch == 1:

            if input_dim == 1024:
                fig = plot_dataset_samples.plot_cifar10_data(cur_samples, batch_labels, title='Epoch {}'.format(str(epoch).zfill(3)), categories=categories, grayscale=True)
            elif input_dim == 3072:
                fig = plot_dataset_samples.plot_cifar10_data(cur_samples, batch_labels, title='Epoch {}'.format(str(epoch).zfill(3)), categories=categories, grayscale=False)
            fig.savefig(output_images_dir + '/epoch_{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close()
    elapsed_time = time.time() - start_time

    print('training time: ' + str(elapsed_time))
    print('')

    error1 = Utilities.rmse(X_train, X_filled)
    print('root mean squared error: ' + str(error1))

    error2 = Utilities.mae(X_train, X_filled)
    print('mean absolute error: ' + str(error2))

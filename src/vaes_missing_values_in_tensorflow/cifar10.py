# WORKS WELL ONLY WITH GRAYSCALED IMAGES! RGB IMAGES HAVE DISSATISFYING RESULTS. #
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10 as cifar10_dataset

from src.utilities.constants import *
from src.utilities.plot_dataset_samples import plot_cifar10_data
from src.utilities.utils import reduce_data, construct_missing_data, get_non_zero_percentage, rmse, mae
from src.utilities.vae_in_tensorflow import vae


def cifar10(
        latent_dim=64,
        epochs=100,
        batch_size='N',
        learning_rate=0.01,
        structured_or_random='structured',
        rgb_or_grayscale='grayscale',
        category=3
):
    missing_value = 0.5

    if rgb_or_grayscale.lower() == 'grayscale':
        output_images_path = output_img_base_path + 'vaes_missing_values_in_tensorflow/cifar10_grayscale'
        logdir = tensorflow_logs_path + 'cifar10_grayscale_vae_missing_values'
        save_path = save_base_path + 'cifar10_grayscale_vae_missing_values'
    else:
        output_images_path = output_img_base_path + 'vaes_missing_values_in_tensorflow/cifar10_rgb'
        logdir = tensorflow_logs_path + 'cifar10_rgb_vae_missing_values'
        save_path = save_base_path + 'cifar10_rgb_vae_missing_values'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # LOAD CIFAR-10 DATASET #
    (X_train, y_train), (X_test, y_test) = cifar10_dataset.load_data()

    X_train = X_train[np.where(y_train == category)[0], :]
    X_test = X_test[np.where(y_test == category)[0], :]

    # merge train and test data together to increase the train dataset size
    X_train = np.concatenate((X_train, X_test), axis=0)  # X_train: N x 3072

    # randomize data
    s = np.random.permutation(X_train.shape[0])
    X_train = X_train[s, :]

    # Reduce data to avoid memory error.
    X_train, _, _ = reduce_data(X_train, X_train.shape[0], 15000)
    X_test, _, _ = reduce_data(X_test, X_test.shape[0], 15000)

    if rgb_or_grayscale.lower() == 'grayscale':
        # convert colored images from 3072 dimensions to 1024 grayscale images
        X_train = np.dot(X_train[:, :, :, :3], [0.299, 0.587, 0.114])
        X_train = np.reshape(X_train, newshape=(-1, 1024))  # X_train: N x 1024
    else:
        # We will flatten the 32x32 images into vectors of size 3072.
        X_train = X_train.reshape((-1, 3072))

    # We will normalize all values between 0 and 1.
    X_train = X_train / 255.

    #####

    # construct data with missing values
    X_train_missing, X_train, _ = construct_missing_data(X=X_train, structured_or_random=structured_or_random)

    print()

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

    x, loss_summ, apply_updates, summary_op, saver, elbo, x_recon_samples = vae(
        batch_size,
        input_dim,
        hidden_encoder_dim,
        hidden_decoder_dim,
        latent_dim,
        lr=learning_rate
    )

    fig = None
    start_index = None
    end_index = None
    batch_labels = None
    cur_samples = None
    cur_elbo = None

    # X_train_masked: array with 0s where the pixels are missing
    # and 1s where the pixels are not missing
    X_train_masked = np.array(X_train_missing)
    X_train_masked[np.where(X_train_masked != missing_value)] = 1
    X_train_masked[np.where(X_train_masked == missing_value)] = 0

    non_zero_percentage = get_non_zero_percentage(X_train_masked)
    print(f'non missing values percentage: {non_zero_percentage} %')

    X_filled = np.array(X_train_missing)

    start_time = time.time()
    with tf.compat.v1.Session() as sess:
        summary_writer = tf.compat.v1.summary.FileWriter(logdir, graph=sess.graph)
        if os.path.isfile(save_path + '/model.ckpt'):
            print('Restoring saved parameters')
            saver.restore(sess, save_path + '/model.ckpt')
        else:
            print('Initializing parameters')
            sess.run(tf.compat.v1.global_variables_initializer())

        print()

        for epoch in range(1, epochs + 1):
            iterations = int(N / batch_size)
            for i in range(1, iterations + 1):
                start_index = (i - 1) * batch_size
                end_index = i * batch_size

                batch_data = X_filled[start_index:end_index, :]
                batch_labels = y_train[start_index:end_index]

                feed_dict = {x: batch_data}
                loss_str, _, summary_str, cur_elbo, cur_samples = sess.run(
                    [loss_summ, apply_updates, summary_op, elbo, x_recon_samples],
                    feed_dict=feed_dict
                )

                masked_batch_data = X_train_masked[start_index:end_index, :]
                cur_samples = np.multiply(masked_batch_data, batch_data) + \
                              np.multiply(1 - masked_batch_data, cur_samples)
                X_filled[start_index:end_index, :] = cur_samples

                summary_writer.add_summary(loss_str, epoch)
                summary_writer.add_summary(summary_str, epoch)

            print(f'Epoch {epoch} | Loss (ELBO): {cur_elbo}')

            if epoch == 1:
                if input_dim == 1024:
                    fig = plot_cifar10_data(X=X_train[start_index:end_index, :], grayscale=True)
                elif input_dim == 3072:
                    fig = plot_cifar10_data(X=X_train[start_index:end_index, :], grayscale=False)
                fig.savefig(f'{output_images_path}/original_data.png', bbox_inches='tight')
                plt.close()

                if input_dim == 1024:
                    fig = plot_cifar10_data(X=X_train_missing[start_index:end_index, :], grayscale=True)
                elif input_dim == 3072:
                    fig = plot_cifar10_data(X=X_train_missing[start_index:end_index, :], grayscale=False)
                fig.savefig(f'{output_images_path}/missing_data.png', bbox_inches='tight')
                plt.close()

            if epoch % 10 == 0 or epoch == 1:

                if input_dim == 1024:
                    fig = plot_cifar10_data(X=cur_samples, title=f'Epoch {str(epoch).zfill(3)}', grayscale=True)
                elif input_dim == 3072:
                    fig = plot_cifar10_data(X=cur_samples, title=f'Epoch {str(epoch).zfill(3)}', grayscale=False)
                fig.savefig(f'{output_images_path}/epoch_{str(epoch).zfill(3)}.png', bbox_inches='tight')
                plt.close()

            if epoch % 2 == 0:
                saver.save(sess, save_path + '/model.ckpt')
    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print()

    print()

    error1 = rmse(X_train, X_filled)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_train, X_filled)
    print(f'mean absolute error: {error2}')

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=./tensorflow_logs/cifar10_rgb_vae_missing_values' OR
    # 'tensorboard --logdir=./tensorflow_logs/cifar10_grayscale_vae_missing_values'.
    # Then open your browser and navigate to -> http://localhost:6006


if __name__ == '__main__':
    cifar10()

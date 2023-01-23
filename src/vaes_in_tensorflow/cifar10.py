# WORKS WELL ONLY WITH GRAYSCALED IMAGES! RGB IMAGES HAVE DISSATISFYING RESULTS. #

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10 as cifar10_dataset

from src.utilities.constants import *
from src.utilities.plot_dataset_samples import plot_cifar10_data
from src.utilities.utils import mae, rmse
from src.utilities.vae_in_tensorflow import vae


def cifar10(latent_dim=64, epochs=100, batch_size='250', learning_rate=0.01, rgb_or_grayscale='grayscale', category=3):
    if rgb_or_grayscale.lower() == 'grayscale':
        output_images_path = output_img_base_path + 'vaes_in_tensorflow/cifar10_grayscale'
        logdir = tensorflow_logs_path + 'cifar10_grayscale_vae'
        save_path = save_base_path + 'cifar10_grayscale_vae'
    else:
        output_images_path = output_img_base_path + 'vaes_in_tensorflow/cifar10_rgb'
        logdir = tensorflow_logs_path + 'cifar10_rgb_vae'
        save_path = save_base_path + 'cifar10_rgb_vae'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # LOAD CIFAR-10 DATASET #
    (X_train, y_train), (X_test, y_test) = cifar10_dataset.load_data()

    # We will normalize all values between 0 and 1,
    # and we will flatten the 32x32 images into vectors of size 3072.

    X_train = X_train[np.where(y_train == category)[0], :]
    X_test = X_test[np.where(y_test == category)[0], :]

    # merge train and test data together to increase the train dataset size
    X_train = np.concatenate((X_train, X_test), axis=0)  # X_train: N x 3072

    # randomize data
    s = np.random.permutation(X_train.shape[0])
    X_train = X_train[s, :]

    if rgb_or_grayscale.lower() == 'grayscale':
        # convert colored images from 3072 dimensions to 1024 grayscale images
        X_train = np.dot(X_train[:, :, :, :3], [0.299, 0.587, 0.114])
        X_train = np.reshape(X_train, newshape=(-1, 1024))  # X_train: N x 1024
    else:
        # We will normalize all values between 0 and 1,
        # and we will flatten the 32x32 images into vectors of size 3072.
        X_train = X_train.reshape((-1, 3072))

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
    cur_samples = None
    cur_elbo = None
    X_recon = np.zeros((N, input_dim))

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

                batch_data = X_train[start_index:end_index, :]

                feed_dict = {x: batch_data}
                loss_str, _, summary_str, cur_elbo, cur_samples = sess.run(
                    [loss_summ, apply_updates, summary_op, elbo, x_recon_samples],
                    feed_dict=feed_dict
                )

                X_recon[start_index:end_index, :] = cur_samples

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

    error1 = rmse(X_train, X_recon)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_train, X_recon)
    print(f'mean absolute error: {error2}')

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../tensorflow_logs/cifar10_rgb_vae' OR
    # 'tensorboard --logdir=../tensorflow_logs/cifar10_grayscale_vae'.
    # Then open your browser and navigate to -> http://localhost:6006


if __name__ == '__main__':
    cifar10()

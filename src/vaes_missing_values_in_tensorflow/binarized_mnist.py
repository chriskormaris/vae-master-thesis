import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.utilities.constants import *
from src.utilities.get_binarized_mnist_dataset import get_binarized_mnist_dataset, get_binarized_mnist_labels, obtain
from src.utilities.plot_utils import plot_images
from src.utilities.utils import construct_missing_data, get_non_zero_percentage, rmse, mae
from src.utilities.vae_in_tensorflow import vae


def binarized_mnist(latent_dim=64, epochs=100, batch_size='250', learning_rate=0.01):
    missing_value = 0.5

    output_images_path = output_img_base_path + 'vaes_missing_values_in_tensorflow/binarized_mnist'

    binarized_dataset_path = '../Binarized_MNIST_dataset'
    logdir = tensorflow_logs_path + 'binarized_mnist_vae_missing_values'
    save_path = save_base_path + 'binarized_mnist_vae_missing_values'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    if not os.path.exists(binarized_dataset_path):
        os.makedirs(binarized_dataset_path)
        obtain(binarized_dataset_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X_test = get_binarized_mnist_dataset(binarized_dataset_path + 'binarized_mnist_test.amat', 'TEST')
    y_test = get_binarized_mnist_labels(binarized_dataset_path + 'binarized_mnist_test_labels.txt', 'TEST')

    # random shuffle the data
    np.random.seed(0)
    s = np.random.permutation(X_test.shape[0])
    X_test = X_test[s, :]
    y_test = y_test[s]

    print()

    #####

    # Reduce data to avoid memory error.
    # X_test, _, _ = reduce_data(X_test, X_test.shape[0], 1000)

    # construct data with missing values
    X_test_missing, X_test, _ = construct_missing_data(X_test)

    print()

    #####

    N = X_test.shape[0]
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

    # plot X_test
    fig = plot_images(X_test, y_test, title='Original Data')
    fig.savefig(f'{output_images_path}/original_data.png', bbox_inches='tight')
    plt.close()

    # plot X_test_missing
    fig = plot_images(X_test_missing, y_test, title='Missing Data')
    fig.savefig(f'{output_images_path}/missing_data.png', bbox_inches='tight')
    plt.close()

    #####

    x, loss_summ, apply_updates, summary_op, saver, elbo, x_recon_samples = vae(
        batch_size,
        input_dim,
        hidden_encoder_dim,
        hidden_decoder_dim,
        latent_dim,
        lr=learning_rate
    )

    cur_samples = None
    batch_labels = None
    cur_elbo = None
    masked_batch_data = None

    # X_test_masked: array with 0s where the pixels are missing
    # and 1s where the pixels are not missing
    X_test_masked = np.array(X_test_missing)
    X_test_masked[np.where(X_test_masked != missing_value)] = 1
    X_test_masked[np.where(X_test_masked == missing_value)] = 0

    non_zero_percentage = get_non_zero_percentage(X_test_masked)
    print(f'non missing values percentage: {non_zero_percentage} %')

    X_filled = np.array(X_test_missing)

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
                batch_labels = y_test[start_index:end_index]

                feed_dict = {x: batch_data}
                loss_str, _, summary_str, cur_elbo, cur_samples = sess.run(
                    [loss_summ, apply_updates, summary_op, elbo, x_recon_samples],
                    feed_dict=feed_dict
                )

                masked_batch_data = X_test_masked[start_index:end_index, :]
                cur_samples = np.multiply(masked_batch_data, batch_data) + \
                              np.multiply(1 - masked_batch_data, cur_samples)
                X_filled[start_index:end_index, :] = cur_samples

                summary_writer.add_summary(loss_str, epoch)
                summary_writer.add_summary(summary_str, epoch)

            print(f'Epoch {epoch} | Loss (ELBO): {cur_elbo}')

            if epoch == 1:
                fig = plot_images(masked_batch_data, batch_labels, title='Masked Data')
                fig.savefig(f'{output_images_path}/masked_data.png', bbox_inches='tight')
                plt.close()

            if epoch % 10 == 0 or epoch == 1:
                fig = plot_images(
                    cur_samples,
                    batch_labels,
                    title=f'Epoch {str(epoch).zfill(3)}'
                )
                fig.savefig(f'{output_images_path}/epoch_{str(epoch).zfill(3)}.png', bbox_inches='tight')
                plt.close()

            if epoch % 2 == 0:
                saver.save(sess, save_path + '/model.ckpt')

    print()
    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print()

    error1 = rmse(X_test, X_filled)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_test, X_filled)
    print(f'mean absolute error: {error2}')

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=./tensorflow_logs/binarized_mnist_vae_missing_values'.
    # Then open your browser and navigate to -> http://localhost:6006


if __name__ == '__main__':
    binarized_mnist()

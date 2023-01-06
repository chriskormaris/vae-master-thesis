import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.Utilities.get_mnist_dataset import get_mnist_dataset

from src.Utilities.constants import *
from src.Utilities.plot_dataset_samples import plot_mnist_or_omniglot_data
from src.Utilities.utils import reduce_data, construct_missing_data, get_non_zero_percentage, rmse, mae
from src.Utilities.vae_in_tensorflow import vae

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings


def mnist(
        latent_dim=64,
        epochs=100,
        batch_size=250,
        learning_rate=0.01,
        structured_or_random='structured',
        digits_or_fashion='digits'
):
    missing_value = 0.5

    if digits_or_fashion == 'digits':
        output_images_path = output_img_base_path + 'VAEsMissingValuesInTensorFlow/mnist'
        logdir = tensorflow_logs_path + 'mnist_vae_missing_values'
        save_path = save_base_path + 'mnist_vae_missing_values'
        dataset_path = mnist_dataset_path
    else:
        output_images_path = output_img_base_path + 'VAEsMissingValuesInTensorFlow/fashion_mnist'
        logdir = tensorflow_logs_path + 'fashion_mnist_vae_missing_values'
        save_path = save_base_path + 'fashion_mnist_vae_missing_values'
        dataset_path = fashion_mnist_dataset_path

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mnist = get_mnist_dataset(dataset_path)

    X_train = mnist[0][0]
    X_train = X_train.reshape(-1, 784)
    y_train = mnist[0][1]

    #####

    # Reduce data to avoid memory error.
    X_train, y_train, _ = reduce_data(X_train, X_train.shape[0], 30000, y=y_train)

    # construct data with missing values
    X_train_missing, X_train, y_train = construct_missing_data(X_train, y_train,
                                                               structured_or_random=structured_or_random)

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

    x, loss_summ, apply_updates, summary_op, saver, elbo, x_recon_samples = vae(
        batch_size,
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

    start_time = time.time()
    with tf.compat.v1.Session() as sess:
        summary_writer = tf.compat.v1.summary.FileWriter(logdir, graph=sess.graph)
        if os.path.isfile(save_path + '/model.ckpt'):
            print('Restoring saved parameters')
            saver.restore(sess, save_path + '/model.ckpt')
        else:
            print('Initializing parameters')
            sess.run(tf.compat.v1.global_variables_initializer())

        print('')

        for epoch in range(1, epochs + 1):
            iterations = int(N / batch_size)
            for i in range(iterations):
                start_index = i * batch_size
                end_index = (i + 1) * batch_size

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

                fig = plot_mnist_or_omniglot_data(
                    masked_batch_data,
                    batch_labels,
                    title='Masked Data'
                )
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

            if epoch == int(epochs / 2):
                saver.save(sess, save_path + '/model.ckpt')
    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print('')

    error1 = rmse(X_train, X_filled)
    print('root mean squared error: ' + str(error1))

    error2 = mae(X_train, X_filled)
    print('mean absolute error: ' + str(error2))

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=./tensorflow_logs/mnist_vae_missing_values'.
    # Then open your browser and navigate to -> http://localhost:6006


if __name__ == '__main__':
    mnist()

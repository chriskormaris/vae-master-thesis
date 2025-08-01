import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist as fashion_mnist_dataset
from keras.datasets import mnist as mnist_dataset

from src import *
from src.utilities import rmse, mae
from src.utilities.plot_utils import plot_images
from src.utilities.vae_in_tensorflow import vae


def mnist(latent_dim=64, epochs=100, batch_size='250', learning_rate=0.01, digits_or_fashion='digits'):
    if digits_or_fashion == 'digits':
        output_images_path = os.path.join(output_img_base_path, 'vaes_in_tensorflow', 'mnist')
        logdir = os.path.join(tensorflow_logs_path, 'mnist_vae')
        save_path = os.path.join(save_base_path, 'mnist_vae')
        mnist_data = mnist_dataset.load_data()
    else:
        output_images_path = os.path.join(output_img_base_path, 'vaes_in_tensorflow', 'fashion_mnist')
        logdir = os.path.join(tensorflow_logs_path, 'fashion_mnist_vae')
        save_path = os.path.join(save_base_path, 'fashion_mnist_vae')
        mnist_data = fashion_mnist_dataset.load_data()

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X_train, y_train = mnist_data[0]

    # We will normalize all values between 0 and 1,
    # and we will flatten the 28x28 images into vectors of size 784.
    X_train = X_train / 255.
    X_train = X_train.reshape((-1, np.prod(X_train.shape[1:])))

    print()

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

    fig = plot_images(X_train, y_train, title='Original Data')
    fig.savefig(os.path.join(output_images_path, 'original_data.png'), bbox_inches='tight')
    plt.close()

    #####

    x, loss_summ, apply_updates, summary_op, saver, elbo, x_recon_samples = vae(
        batch_size,
        input_dim,
        hidden_encoder_dim,
        hidden_decoder_dim,
        latent_dim,
        learning_rate=learning_rate
    )

    cur_samples = None
    batch_labels = None
    cur_elbo = None
    X_recon = np.zeros((N, input_dim))

    start_time = time.time()
    with tf.compat.v1.Session() as sess:
        summary_writer = tf.compat.v1.summary.FileWriter(logdir, graph=sess.graph)
        if os.path.isfile(os.path.join(save_path, 'model.ckpt')):
            print('Restoring saved parameters')
            saver.restore(sess, os.path.join(save_path, 'model.ckpt'))
        else:
            print('Initializing parameters')
            sess.run(tf.compat.v1.global_variables_initializer())

        print()

        for epoch in range(1, epochs + 1):
            iterations = int(N / batch_size)
            for i in range(1, iterations + 1):
                start_index = (i - 1) * batch_size
                end_index = i * batch_size

                # batch_data, batch_labels = mnist.train.next_batch(batch_size=batch_size)
                # ALTERNATIVE
                batch_data = X_train[start_index:end_index, :]
                batch_labels = y_train[start_index:end_index]

                feed_dict = {x: batch_data}
                loss_str, _, summary_str, cur_elbo, cur_samples = sess.run(
                    [loss_summ, apply_updates, summary_op, elbo, x_recon_samples],
                    feed_dict=feed_dict
                )

                X_recon[start_index:end_index, :] = cur_samples

                summary_writer.add_summary(loss_str, epoch)
                summary_writer.add_summary(summary_str, epoch)

            print(f'Epoch {epoch} | Loss (ELBO): {cur_elbo}')

            if epoch % 10 == 0 or epoch == 1:
                fig = plot_images(
                    cur_samples,
                    batch_labels,
                    title=f'Epoch {str(epoch).zfill(3)}'
                )
                fig.savefig(os.path.join(output_images_path, f'epoch_{str(epoch).zfill(3)}.png'), bbox_inches='tight')
                plt.close()

            if epoch % 2 == 0:
                saver.save(sess, os.path.join(save_path, 'model.ckpt'))
    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print()

    error1 = rmse(X_train, X_recon)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_train, X_recon)
    print(f'mean absolute error: {error2}')

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../tensorflow_logs/mnist_vae' OR
    # 'tensorboard --logdir=../tensorflow_logs/fashion_mnist_vae'.
    # Then open your browser and navigate to -> http://localhost:6006


if __name__ == '__main__':
    mnist()

# WORKS BETTER WITH THE ENGLISH ALPHABET! THE GREEK ALPHABET HAS VERY FEW EXAMPLES. #

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.Utilities.constants import *
from src.Utilities.get_omniglot_dataset import get_omniglot_dataset
from src.Utilities.plot_dataset_samples import plot_mnist_or_omniglot_data
from src.Utilities.utils import rmse, mae
from src.Utilities.vae_in_tensorflow import vae

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings


def omniglot(latent_dim=64, epochs=100, batch_size=250, learning_rate=0.01, language='English'):
    if language.lower() == 'greek':
        output_images_path = output_img_base_path + 'VAEsInTensorFlow/omniglot_greek'
        logdir = 'tensorflow_logs/omniglot_greek_vae'
        save_path = 'save/omniglot_greek_vae'
        alphabet = 20
    else:
        output_images_path = output_img_base_path + 'VAEsInTensorFlow/omniglot_english'
        logdir = 'tensorflow_logs/omniglot_english_vae'
        save_path = 'save/omniglot_english_vae'
        alphabet = 31

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # LOAD OMNIGLOT DATASET #
    X_train, y_train = get_omniglot_dataset(
        omniglot_dataset_path + '/chardata.mat',
        train_or_test='train',
        alphabet=alphabet,
        binarize=True
    )
    X_test, y_test = get_omniglot_dataset(
        omniglot_dataset_path + '/chardata.mat',
        train_or_test='test',
        alphabet=alphabet,
        binarize=True
    )

    X_merged = np.concatenate((X_train, X_test), axis=0)
    y_merged = np.concatenate((y_train, y_test), axis=0)

    #####

    N = X_merged.shape[0]
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

    fig = plot_mnist_or_omniglot_data(X_merged, y_merged, categories=list(range(1, 11)), title='Original Data')
    fig.savefig(output_images_path + '/original_data_characters_1-10.png', bbox_inches='tight')
    plt.close()
    fig = plot_mnist_or_omniglot_data(X_merged, y_merged, categories=list(range(11, 21)), title='Original Data')
    fig.savefig(output_images_path + '/original_data_characters_11-20.png', bbox_inches='tight')
    plt.close()
    if language.lower() == 'greek':
        fig = plot_mnist_or_omniglot_data(X_merged, y_merged, categories=list(range(21, 25)), title='Original Data')
        fig.savefig(output_images_path + '/original_data_characters_21-24.png', bbox_inches='tight')
    else:
        fig = plot_mnist_or_omniglot_data(X_merged, y_merged, categories=list(range(21, 27)), title='Original Data')
        fig.savefig(output_images_path + '/original_data_characters_21-26.png', bbox_inches='tight')
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

    batch_labels = None
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

        print('')

        for epoch in range(1, epochs + 1):
            iterations = int(N / batch_size)
            for i in range(iterations):
                start_index = i * batch_size
                end_index = (i + 1) * batch_size

                batch_data = X_merged[start_index:end_index, :]
                batch_labels = y_merged[start_index:end_index]

                feed_dict = {x: batch_data}
                loss_str, _, summary_str, cur_elbo, cur_samples = sess.run(
                    [loss_summ, apply_updates, summary_op, elbo, x_recon_samples],
                    feed_dict=feed_dict
                )

                X_recon[start_index:end_index] = cur_samples

                summary_writer.add_summary(loss_str, epoch)
                summary_writer.add_summary(summary_str, epoch)

            print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))

            if epoch % 10 == 0 or epoch == 1:

                fig = plot_mnist_or_omniglot_data(
                    cur_samples,
                    batch_labels,
                    categories=list(range(1, 11)), n=5, title='Epoch {}'.format(str(epoch).zfill(3)),
                    mnist_or_omniglot='omniglot'
                )
                fig.savefig(
                    output_images_path + '/epoch_{}_characters_1-10.png'.format(str(epoch).zfill(3)),
                    bbox_inches='tight'
                )
                plt.close()
                fig = plot_mnist_or_omniglot_data(
                    cur_samples,
                    batch_labels,
                    categories=list(range(11, 21)), n=5, title='Epoch {}'.format(str(epoch).zfill(3)),
                    mnist_or_omniglot='omniglot'
                )
                fig.savefig(
                    output_images_path + '/epoch_{}_characters_11-20.png'.format(str(epoch).zfill(3)),
                    bbox_inches='tight'
                )
                plt.close()
                if language.lower() == 'greek':
                    fig = plot_mnist_or_omniglot_data(
                        cur_samples,
                        batch_labels,
                        categories=list(range(21, 25)), n=5, title='Epoch {}'.format(str(epoch).zfill(3)),
                        mnist_or_omniglot='omniglot'
                    )
                    fig.savefig(
                        output_images_path + '/epoch_{}_characters_21-24.png'.format(str(epoch).zfill(3)),
                        bbox_inches='tight'
                    )
                else:
                    fig = plot_mnist_or_omniglot_data(
                        cur_samples,
                        batch_labels,
                        categories=list(range(21, 27)),
                        n=5,
                        title='Epoch {}'.format(str(epoch).zfill(3)),
                        mnist_or_omniglot='omniglot'
                    )
                    fig.savefig(
                        output_images_path + '/epoch_{}_characters_21-26.png'.format(str(epoch).zfill(3)),
                        bbox_inches='tight'
                    )
                plt.close()

            if epoch == int(epochs / 2):
                saver.save(sess, save_path + '/model.ckpt')
    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print('')

    error1 = rmse(X_merged, X_recon)
    print('root mean squared error: ' + str(error1))

    error2 = mae(X_merged, X_recon)
    print('mean absolute error: ' + str(error2))

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../tensorflow_logs/omniglot_greek_vae' OR
    # 'tensorboard --logdir=../tensorflow_logs/omniglot_english_vae'.
    # Then open your browser and navigate to -> http://localhost:6006


if __name__ == '__main__':
    omniglot()

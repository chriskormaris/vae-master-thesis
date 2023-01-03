import inspect
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from __init__ import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings

__author__ = 'c.kormaris'


#####

# MAIN #

if __name__ == '__main__':

    binarized_mnist_dataset_dir = '../Binarized_MNIST_dataset'

    output_images_dir = './output_images/VAEsInTensorFlow/binarized_mnist'
    log_dir = './tensorflow_logs/binarized_mnist_vae'
    save_dir = './save/binarized_mnist_vae'

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    if not os.path.exists(binarized_mnist_dataset_dir):
        os.makedirs(binarized_mnist_dataset_dir)
        get_binarized_mnist_dataset.obtain(binarized_mnist_dataset_dir)

    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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

    x, loss_summ, apply_updates, summary_op, saver, elbo, x_recon_samples = \
        vae_in_tensorflow.vae(batch_size, input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, lr=learning_rate)

    cur_samples = None
    batch_labels = None
    cur_elbo = None
    X_recon = np.zeros((N, input_dim))

    start_time = time.time()
    with tf.compat.v1.Session() as sess:
        summary_writer = tf.compat.v1.summary.FileWriter(log_dir, graph=sess.graph)
        if os.path.isfile(save_dir + '/model.ckpt'):
            print('Restoring saved parameters')
            saver.restore(sess, save_dir + '/model.ckpt')
        else:
            print('Initializing parameters')
            sess.run(tf.compat.v1.global_variables_initializer())

        print('')

        for epoch in range(1, epochs + 1):
            iterations = int(N / batch_size)
            for i in range(iterations):
                start_index = i * batch_size
                end_index = (i + 1) * batch_size

                batch_data = X_test[start_index:end_index, :]
                batch_labels = y_test[start_index:end_index]

                feed_dict = {x: batch_data}
                loss_str, _, summary_str, cur_elbo, cur_samples = sess.run(
                    [loss_summ, apply_updates, summary_op, elbo, x_recon_samples], feed_dict=feed_dict)

                X_recon[start_index:end_index] = cur_samples

                summary_writer.add_summary(loss_str, epoch)
                summary_writer.add_summary(summary_str, epoch)

            print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))

            if epoch % 10 == 0 or epoch == 1:

                fig = plot_dataset_samples.plot_mnist_or_omniglot_data(cur_samples, batch_labels, title='Epoch {}'.format(str(epoch).zfill(3)))
                fig.savefig(output_images_dir + '/epoch_{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
                plt.close()

            if epoch == int(epochs / 2):
                save_path = saver.save(sess, save_dir + '/model.ckpt')

    print('')
    elapsed_time = time.time() - start_time

    print('training time: ' + str(elapsed_time))
    print('')

    error1 = Utilities.rmse(X_test, X_recon)
    print('root mean squared error: ' + str(error1))

    error2 = Utilities.mae(X_test, X_recon)
    print('mean absolute error: ' + str(error2))


    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../tensorflow_logs/binarized_mnist_vae'.
    # Then open your browser and navigate to -> http://localhost:6006

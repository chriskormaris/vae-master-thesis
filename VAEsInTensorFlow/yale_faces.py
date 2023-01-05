import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Utilities.get_yale_faces_dataset import get_yale_faces_dataset
from Utilities.plot_dataset_samples import plot_yale_faces
from Utilities.utils import rmse, mae
from Utilities.vae_in_tensorflow import vae

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings


def yale_faces(latent_dim=64, epochs=100, batch_size=250, learning_rate=0.01):
    yale_faces_dataset_dir = '../YALE_dataset/CroppedYale'

    output_images_dir = './output_images/VAEsInTensorFlow/yale_faces'
    log_dir = './tensorflow_logs/yale_faces_vae'
    save_dir = './save/yale_faces_vae'

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # LOAD YALE FACES DATASET #
    print('Getting YALE faces dataset...')
    X, y = get_yale_faces_dataset(yale_faces_dataset_dir)

    #####

    N = X.shape[0]
    input_dim = 32256  # D
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

    for i in range(0, 38, 10):
        if i <= 20:
            fig = plot_yale_faces(
                X,
                y,
                categories=list(range(i, i + 10)),
                title='Original Faces',
                show_plot=False
            )
            fig.savefig(output_images_dir + '/original_faces_' + str(i + 1) + '-' + str(i + 10) + '.png')
            plt.close()
        else:
            fig = plot_yale_faces(
                X,
                y,
                categories=list(range(30, 38)),
                title='Original Faces',
                show_plot=False
            )
            fig.savefig(output_images_dir + '/original_faces_' + str(31) + '-' + str(38) + '.png')
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

                batch_data = X[start_index:end_index, :]
                batch_labels = y[start_index:end_index]

                feed_dict = {x: batch_data}
                loss_str, _, summary_str, cur_elbo, cur_samples = sess.run(
                    [loss_summ, apply_updates, summary_op, elbo, x_recon_samples], feed_dict=feed_dict)

                X_recon[start_index:end_index] = cur_samples

                summary_writer.add_summary(loss_str, epoch)
                summary_writer.add_summary(summary_str, epoch)

            print('Epoch {0} | Loss (ELBO): {1}'.format(epoch, cur_elbo))

            if epoch % 10 == 0 or epoch == 1:
                fig = plot_yale_faces(
                    cur_samples,
                    batch_labels,
                    categories=list(range(10)), title='Epoch {}'.format(str(epoch).zfill(3))
                )
                fig.savefig(output_images_dir + '/epoch_{}_faces.png'.format(str(epoch).zfill(3)))
                plt.close()

            if epoch == int(epochs / 2):
                saver.save(sess, save_dir + '/model.ckpt')
    elapsed_time = time.time() - start_time

    print('training time: ' + str(elapsed_time))
    print('')

    error1 = rmse(X, X_recon)
    print('root mean squared error: ' + str(error1))

    error2 = mae(X, X_recon)
    print('mean absolute error: ' + str(error2))

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../tensorflow_logs/yale_faces_vae'.
    # Then open your browser and navigate to -> http://localhost:6006

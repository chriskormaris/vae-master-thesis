import os
import time

import numpy as np
from keras.callbacks import TensorBoard

from src import *
from src.utilities import rmse, mae
from src.utilities.get_binarized_mnist_dataset import get_binarized_mnist_dataset, obtain
from src.utilities.plot_utils import plot_original_vs_reconstructed_data
from src.utilities.vae_in_keras import vae


def binarized_mnist(latent_dim=64, epochs=100, batch_size='N', learning_rate=0.001):
    input_dim = 784
    output_images_path = os.path.join(output_img_base_path, 'vaes_in_keras')

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    encoder, decoder, autoencoder = vae(input_dim, latent_dim, learning_rate)

    # Let's prepare our input data.
    # We're using MNIST digits, and we're discarding the labels
    # (since we're only interested in encoding/decoding the input images).

    if not os.path.exists(binarized_dataset_path):
        os.makedirs(binarized_dataset_path)
        obtain(binarized_dataset_path)

    X_train = get_binarized_mnist_dataset(os.path.join(binarized_dataset_path, 'binarized_mnist_train.amat'), 'TRAIN')
    X_test = get_binarized_mnist_dataset(os.path.join(binarized_dataset_path, 'binarized_mnist_test.amat'), 'TEST')

    # randomize data
    s = np.random.permutation(X_train.shape[0])
    X_train = X_train[s, :]
    s = np.random.permutation(X_test.shape[0])
    X_test = X_test[s, :]

    # Now let's train our autoencoder for a given number of epochs. #
    if batch_size == 'N':
        batch_size = X_train.shape[0]
    else:
        batch_size = int(batch_size)

    start_time = time.time()
    autoencoder.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_test, X_test),
        callbacks=[TensorBoard(histogram_freq=0, write_graph=False)]
    )
    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print()

    # We can try to visualize the reconstructed inputs and the encoded representations.
    # We will use Matplotlib.

    # encode and decode some digits
    # note that we take them from the "test" set
    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    print(f'encoded_imgs mean: {encoded_imgs.mean()}')

    fig = plot_original_vs_reconstructed_data(X_test, decoded_imgs, grayscale=True)
    fig.savefig(os.path.join(output_images_path, 'binarized_mnist.png'))

    print()

    error1 = rmse(X_test, decoded_imgs)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_test, decoded_imgs)
    print(f'mean absolute error: {error2}')

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../keras_logs/binarized_mnist'.
    # Then open your browser ang navigate to: http://localhost:6006


if __name__ == '__main__':
    binarized_mnist()

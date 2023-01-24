import os
import time

import numpy as np
from keras.callbacks import TensorBoard
from keras.datasets import fashion_mnist as fashion_mnist_dataset
from keras.datasets import mnist as mnist_dataset

from src.utilities.constants import *
from src.utilities.plot_utils import plot_original_vs_reconstructed_data
from src.utilities.utils import rmse, mae
from src.utilities.vae_in_keras import vae


def mnist(latent_dim=64, epochs=100, batch_size='250', digits_or_fashion='digits'):
    input_dim = 784
    output_images_path = output_img_base_path + 'vaes_in_keras'

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    if digits_or_fashion == 'digits':
        mnist_data = mnist_dataset.load_data()
    else:
        mnist_data = fashion_mnist_dataset.load_data()

    encoder, decoder, autoencoder = vae(input_dim, latent_dim)

    # Let's prepare our input data.
    # We're using MNIST images, and we're discarding the labels
    # (since we're only interested in encoding/decoding the input images).

    (X_train, y_train), (X_test, y_test) = mnist_data

    # We will normalize all values between 0 and 1,
    # and we will flatten the 28x28 images into vectors of size 784.
    X_train = X_train / 255.
    X_test = X_test / 255.
    X_train = X_train.reshape((-1, np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((-1, np.prod(X_test.shape[1:])))

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
    fig.savefig(f'{output_images_path}/mnist.png')

    print()

    error1 = rmse(X_test, decoded_imgs)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_test, decoded_imgs)
    print(f'mean absolute error: {error2}')

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../keras_logs/mnist'.
    # Then open your browser ang navigate to: http://localhost:6006


if __name__ == '__main__':
    mnist()

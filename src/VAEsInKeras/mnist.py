import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard

from src.Utilities.constants import *
from src.Utilities.get_mnist_dataset import get_mnist_dataset
from src.Utilities.utils import rmse, mae
from src.Utilities.vae_in_keras import vae

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings


def mnist(latent_dim=64, epochs=100, batch_size='250', digits_or_fashion='digits'):
    input_dim = 784

    if digits_or_fashion == 'digits':
        dataset_path = mnist_dataset_path
    else:
        dataset_path = fashion_mnist_dataset_path

    encoder, decoder, autoencoder = vae(input_dim, latent_dim)

    # Let's prepare our input data.
    # We're using MNIST images, and we're discarding the labels
    # (since we're only interested in encoding/decoding the input images).

    mnist = get_mnist_dataset(dataset_path)

    X_train = mnist[0][0]
    X_test = mnist[1][0]

    # We will normalize all values between 0 and 1
    # and we will flatten the 28x28 images into vectors of size 784.

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

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
    print('')

    # We can try to visualize the reconstructed inputs and the encoded representations.
    # We will use Matplotlib.

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    print(f'encoded_imgs mean: {encoded_imgs.mean()}')

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    print('')

    error1 = rmse(X_test, decoded_imgs)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_test, decoded_imgs)
    print(f'mean absolute error: {error2}')

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../keras_logs/mnist'.
    # Then open your browser ang navigate to: http://localhost:6006


if __name__ == '__main__':
    mnist()

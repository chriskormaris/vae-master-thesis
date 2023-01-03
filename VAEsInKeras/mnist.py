import inspect
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from keras.datasets.mnist import load_data

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from __init__ import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings

input_dim = 784
latent_dim = int(sys.argv[1])

digitsOrFashion = sys.argv[4]
# digitsOrFashion = 'digits'
# digitsOrFashion = 'fashion'

if digitsOrFashion == 'digits':
    mnist_dataset_dir = '../MNIST_dataset'
    log_dir = './keras_logs/mnist'
else:
    mnist_dataset_dir = '../FASHION_MNIST_dataset'
    log_dir = './keras_logs/fashion_mnist'


if __name__ == '__main__':

    encoder, decoder, autoencoder = vae_in_keras.vae(input_dim, latent_dim)

    # Let's prepare our input data.
    # We're using MNIST images, and we're discarding the labels
    # (since we're only interested in encoding/decoding the input images).

    mnist = load_data(mnist_dataset_dir)

    X_train = mnist[0][0]
    X_test = mnist[1][0]

    # We will normalize all values between 0 and 1
    # and we will flatten the 28x28 images into vectors of size 784.

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

    # Now let's train our autoencoder for a given number of epochs. #
    epochs = int(sys.argv[2])
    batch_size = sys.argv[3]
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
        callbacks=[TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)]
    )
    elapsed_time = time.time() - start_time

    print('training time: ' + str(elapsed_time))
    print('')

    # We can try to visualize the reconstructed inputs and the encoded representations.
    # We will use Matplotlib.

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    print('encoded_imgs mean: ' + str(encoded_imgs.mean()))

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

    error1 = Utilities.rmse(X_test, decoded_imgs)
    print('root mean squared error: ' + str(error1))

    error2 = Utilities.mae(X_test, decoded_imgs)
    print('mean absolute error: ' + str(error2))

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../keras_logs/mnist'.
    # Then open your browser ang navigate to: http://localhost:6006

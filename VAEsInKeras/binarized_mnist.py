import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard

from Utilities.get_binarized_mnist_dataset import get_binarized_mnist_dataset, get_binarized_mnist_labels, obtain
from Utilities.utils import rmse, mae
from Utilities.vae_in_keras import vae

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings


def binarized_mnist(latent_dim=64, epochs=100, batch_size=250):
    input_dim = 784

    binarized_mnist_dataset_dir = '../Binarized_MNIST_dataset'
    log_dir = './keras_logs/binarized_mnist'

    encoder, decoder, autoencoder = vae(input_dim, latent_dim)

    # Let's prepare our input data.
    # We're using MNIST digits, and we're discarding the labels
    # (since we're only interested in encoding/decoding the input images).

    if not os.path.exists(binarized_mnist_dataset_dir):
        os.makedirs(binarized_mnist_dataset_dir)
        obtain(binarized_mnist_dataset_dir)

    X_train = get_binarized_mnist_dataset(binarized_mnist_dataset_dir + '/binarized_mnist_train.amat', 'TRAIN')
    y_train = get_binarized_mnist_labels(binarized_mnist_dataset_dir + '/binarized_mnist_train_labels.txt', 'TEST')
    X_test = get_binarized_mnist_dataset(binarized_mnist_dataset_dir + '/binarized_mnist_test.amat', 'TEST')

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
        y_train,
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

    error1 = rmse(X_test, decoded_imgs)
    print('root mean squared error: ' + str(error1))

    error2 = mae(X_test, decoded_imgs)
    print('mean absolute error: ' + str(error2))

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../keras_logs/binarized_mnist'.
    # Then open your browser ang navigate to: http://localhost:6006

import time

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard

from src.utilities.constants import *
from src.utilities.get_omniglot_dataset import get_omniglot_dataset
from src.utilities.utils import rmse, mae
from src.utilities.vae_in_keras import vae


def omniglot(latent_dim=64, epochs=100, batch_size='250', language='English'):
    input_dim = 784

    if language.lower() == 'greek':
        alphabet = 20
    else:
        alphabet = 31

    encoder, decoder, autoencoder = vae(input_dim, latent_dim)

    # Let's prepare our input data.
    # We're using OMNIGLOT images, and we're discarding the labels
    # (since we're only interested in encoding/decoding the input images).

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

    # Now let's train our autoencoder for a given number of epochs. #
    if batch_size == 'N':
        batch_size = X_merged.shape[0]
    else:
        batch_size = int(batch_size)

    start_time = time.time()
    autoencoder.fit(
        X_merged,
        X_merged,
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

    print()

    error1 = rmse(X_test, decoded_imgs)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_test, decoded_imgs)
    print(f'mean absolute error: {error2}')

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../keras_logs/omniglot_english'.
    # Then open your browser ang navigate to: http://localhost:6006


if __name__ == '__main__':
    omniglot()

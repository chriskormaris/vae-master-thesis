import inspect
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from __init__ import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings

input_dim = 32256
latent_dim = int(sys.argv[1])
yale_faces_dataset_dir = '../YALE_dataset/CroppedYale'

log_dir = './keras_logs/yale_faces'


if __name__ == '__main__':

    encoder, decoder, autoencoder = vae_in_keras.vae(input_dim, latent_dim)

    # Let's prepare our input data.
    # We're using Faces images, and we're discarding the labels
    # (since we're only interested in encoding/decoding the input images).

    # LOAD OMNIGLOT DATASET #
    print('Getting YALE faces dataset...')
    X, _ = get_yale_faces_dataset.get_yale_faces_dataset(yale_faces_dataset_dir)

    # Now let's train our autoencoder for a given number of epochs. #
    epochs = int(sys.argv[2])
    batch_size = sys.argv[3]
    if batch_size == 'N':
        batch_size = X.shape[0]
    else:
        batch_size = int(batch_size)

    start_time = time.time()
    autoencoder.fit(X, X,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X, X),
                    callbacks=[TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)])
    elapsed_time = time.time() - start_time

    print('training time: ' + str(elapsed_time))
    print('')

    # We can try to visualize the reconstructed inputs and the encoded representations.
    # We will use Matplotlib.

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(X)
    decoded_imgs = decoder.predict(encoded_imgs)

    print('encoded_imgs mean: ' + str(encoded_imgs.mean()))

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(168, 192))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(168, 192))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


    def rmse(X, X_recon):
        return np.sqrt(np.mean(np.square(X - X_recon)))


    print('')

    error1 = Utilities.rmse(X, decoded_imgs)
    print('root mean squared error: ' + str(error1))

    error2 = Utilities.mae(X, decoded_imgs)
    print('mean absolute error: ' + str(error2))

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../keras_logs/omniglot_english'.
    # Then open your browser ang navigate to: http://localhost:6006

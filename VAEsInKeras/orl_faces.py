import os
import time

import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

from Utilities.Utilities import rmse, mae
from Utilities.get_orl_faces_dataset import get_orl_faces_dataset
from Utilities.vae_in_keras import vae

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings


def orl_faces(latent_dim=64, epochs=100, batch_size=250, learning_rate=0.01):
    input_dim = 10304
    faces_dataset_dir = '../ORL_Face_dataset'

    log_dir = './keras_logs/orl_faces'

    encoder, decoder, autoencoder = vae(input_dim, latent_dim)

    # Let's prepare our input data.
    # We're using Faces images, and we're discarding the labels
    # (since we're only interested in encoding/decoding the input images).

    # LOAD ORL FACES DATASET #
    X, y = get_orl_faces_dataset(faces_dataset_dir)

    # Now let's train our autoencoder for a given number of epochs. #
    if batch_size == 'N':
        batch_size = X.shape[0]
    else:
        batch_size = int(batch_size)

    start_time = time.time()
    autoencoder.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X, X),
        callbacks=[TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)]
    )
    elapsed_time = time.time() - start_time

    print('training time: ' + str(elapsed_time))
    print('')
    # We can try to visualize the reconstructed inputs and the encoded representations.
    # We will use Matplotlib.

    # encode and decode some faces
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(X)
    decoded_imgs = decoder.predict(encoded_imgs)

    print('encoded_imgs mean: ' + str(encoded_imgs.mean()))

    n = 10  # how many faces we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(92, 112))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(92, 112))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    print('')

    error1 = rmse(X, decoded_imgs)
    print('root mean squared error: ' + str(error1))

    error2 = mae(X, decoded_imgs)
    print('mean absolute error: ' + str(error2))

    # TENSORBOARD
    # Open a console and run 'tensorboard --logdir=../keras_logs/omniglot_english'.
    # Then open your browser ang navigate to: http://localhost:6006

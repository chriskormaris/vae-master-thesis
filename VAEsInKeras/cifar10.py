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

RGBOrGrayscale = sys.argv[4]

if RGBOrGrayscale.lower() == 'rgb':
    input_dim = 3072
    log_dir = './keras_logs/cifar10_rgb'
else:
    input_dim = 1024
    log_dir = './keras_logs/cifar10_grayscale'

latent_dim = int(sys.argv[1])
cifar10_dataset_dir = '../CIFAR_dataset/CIFAR-10'

encoder, decoder, autoencoder = vae_in_keras.vae(input_dim, latent_dim)

# Let's prepare our input data.
# We're using CIFAR-10 images, and we're discarding the labels
# (since we're only interested in encoding/decoding the input images).

# LOAD CIFAR-10 DATASET #
(X_train, y_train), (X_test, y_test) = get_cifar10_dataset.get_cifar10_dataset(cifar10_dataset_dir)
print('')

# reduce data to avoid Memory error
X_train = X_train[:10000, :] 
y_train = y_train[:10000]
X_test = X_test[:5000, :]
y_test = y_test[:5000]

if RGBOrGrayscale.lower() == 'grayscale':
    # convert colored images from 3072 dimensions to 1024 grayscale images
    X_train = np.dot(X_train[:, :, :, :3], [0.299, 0.587, 0.114])
    X_train = np.reshape(X_train, newshape=(-1, 1024))  # X_train: N x 1024
    X_test = np.dot(X_test[:, :, :, :3], [0.299, 0.587, 0.114])
    X_test = np.reshape(X_test, newshape=(-1, 1024))  # X_test: N x 1024
else:
    # We will normalize all values between 0 and 1
    # and we will flatten the 32x32 images into vectors of size 3072.
    X_train = X_train.reshape((len(X_train), input_dim))
    X_test = X_test.reshape((len(X_test), input_dim))

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# We will normalize all values between 0 and 1
# and we will flatten the 32x32 images into vectors of size input_dim.

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), input_dim))
X_test = X_test.reshape((len(X_test), input_dim))


# Now let's train our autoencoder for a given number of epochs. #
epochs = int(sys.argv[2])
batch_size = sys.argv[3]
if batch_size == 'N':
    batch_size = X_train.shape[0]
else:
    batch_size = int(batch_size)

start_time = time.time()
autoencoder.fit(X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)])
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
    if RGBOrGrayscale.lower() == 'rgb':
        plt.imshow(X_test[i].reshape(32, 32, 3))
    else:
        plt.imshow(X_test[i].reshape(32, 32))
        plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    if RGBOrGrayscale.lower() == 'rgb':
        plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
    else:
        plt.imshow(decoded_imgs[i].reshape(32, 32))
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
# Open a console and run 'tensorboard --logdir=../keras_logs/cifar10_rgb' OR
# 'tensorboard --logdir=../keras_logs/cifar10_grayscale'.
# Then open your browser ang navigate to: http://localhost:6006

from keras.layers import Input, Dense
from keras.models import Model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tensorflow warnings

__author__ = 'c.kormaris'


#####


def vae(input_dim, latent_dim):
    # latent_dim: this is the size of our encoded representations
    # latent_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    # latent_dim = 64  # 64 floats -> compression of factor 12.25, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(input_dim,))

    # 'encoded' is the encoded representation of the input
    encoded = Dense(latent_dim, activation='relu')(input_img)
    # 'decoded' is the lossy reconstruction of the input
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # ALTERNATIVE
    # 'encoded' is the encoded representation of the input
    #encoded = Dense(latent_dim, activation='sigmoid')(input_img)
    # 'decoded' is the lossy reconstruction of the input
    #decoded = Dense(input_dim)(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # Let's also create a separate encoder model. #

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # As well as the decoder model. #

    # create a placeholder for an encoded latent_dim input
    encoded_input = Input(shape=(latent_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # Now let's train our autoencoder to reconstruct the input images.
    # First, we'll configure our model to use a per-pixel binary crossentropy loss,
    # and the Adadelta optimizer.

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')  # Works best!
    #autoencoder.compile(optimizer='adagrad', loss='binary_crossentropy')
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    #autoencoder.compile(optimizer='adamax', loss='binary_crossentropy')
    #autoencoder.compile(optimizer='nadam', loss='binary_crossentropy')
    # autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
    #autoencoder.compile(optimizer='sgd', loss='binary_crossentropy')
    #autoencoder.compile(optimizer='tfoptimizer', loss='binary_crossentropy')

    return encoder, decoder, autoencoder

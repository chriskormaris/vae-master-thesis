# Based on the Vanilla VAE with TensorFlow from this url:
# https://github.com/wiseodd/generative-models

import tensorflow as tf


#####

# HELPER FUNCTIONS #

def initialize_weight_variable(shape, name=''):
    initial = tf.random.truncated_normal(shape, stddev=0.001, name='truncated_normal_' + name)
    return tf.Variable(initial, name=name)


def initialize_bias_variable(shape, name=''):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name=name)


#####


# VARIATIONAL AUTOENCODER IMPLEMENTATION IN TENSORFLOW #

def vae(batch_size, input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, lr=0.01):
    # Reset the default graph
    # IMPORTANT: WE NEED THIS to rerun the TensorFlow operation
    tf.compat.v1.reset_default_graph()

    tf.compat.v1.disable_eager_execution()
    # Input placeholder
    with tf.name_scope('input_data'):
        x = tf.compat.v1.placeholder('float', [batch_size, input_dim], name='X')

        # reshape images to the appropriate format and write to summary
        if input_dim == 784:  # for MNIST or OMNIGLOT datasets
            tf.compat.v1.summary.image('image_data', tf.reshape(x, shape=[-1, 28, 28, 1]))
        elif input_dim == 1024:  # for CIFAR-10 Grayscale dataset
            tf.compat.v1.summary.image('image_data', tf.reshape(x, shape=[-1, 32, 32, 1]))
        elif input_dim == 3072:  # for CIFAR-10 RGB dataset
            tf.compat.v1.summary.image('image_data', tf.reshape(x, shape=[-1, 32, 32, 3]))
        elif input_dim == 10304:  # for ORL Faces dataset
            tf.compat.v1.summary.image(
                'image_data',
                tf.transpose(tf.reshape(x, shape=[-1, 92, 112, 1]), perm=[0, 2, 1, 3])
            )
        elif input_dim == 32256:  # for Yale Faces dataset
            tf.compat.v1.summary.image(
                'image_data',
                tf.transpose(tf.reshape(x, shape=[-1, 192, 168, 1]), perm=[0, 2, 1, 3])
            )

    # ============================== Q(Z|X) = Q(Z) - Encoder NN ============================== #

    # The encoder is a neural network with 2 hidden layers.
    with tf.name_scope('encoder'):
        with tf.name_scope('Thetas'):
            # theta1: M1 x D
            theta1 = initialize_weight_variable([hidden_encoder_dim, input_dim], name='theta1')

            # bias_theta1: 1 x M1
            bias_theta1 = initialize_bias_variable([hidden_encoder_dim], name='bias_theta1')

            # theta_mu: Z_dim x M1
            theta_mu = initialize_weight_variable([latent_dim, hidden_encoder_dim], name='theta_mu')

            # bias_theta_mu: 1 x Z_dim
            bias_theta_mu = initialize_bias_variable([latent_dim], name='bias_theta_mu')

            # theta_logvar: Z_dim x M1
            theta_logvar = initialize_weight_variable([latent_dim, hidden_encoder_dim], name='theta_logvar')

            # bias_theta_logvar: 1 X Z_dim
            bias_theta_logvar = initialize_bias_variable([latent_dim], name='bias_theta_logvar')

        with tf.name_scope('hidden_layer1'):
            # Hidden layer 1 activation function of the encoder
            # hidden_layer_encoder: N x M1
            # RELU
            hidden_layer_encoder = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, tf.transpose(theta1)), bias_theta1))
            # SIGMOID
            # hidden_layer_encoder = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, tf.transpose(theta1)), bias_theta1))

        with tf.name_scope('hidden_layer2_mu'):
            # mu_encoder: N x Z_dim
            mu_encoder = tf.nn.bias_add(tf.matmul(hidden_layer_encoder, tf.transpose(theta_mu)), bias_theta_mu)

        with tf.name_scope('hidden_layer2_logvar'):
            # the log sigma^2 of the encoder
            # logvar_encoder: N x Z_dim
            logvar_encoder = tf.nn.bias_add(
                tf.matmul(hidden_layer_encoder, tf.transpose(theta_logvar)),
                bias_theta_logvar
            )

        with tf.name_scope('sample_E'):
            # Sample epsilon
            # epsilon: N x Z_dim
            epsilon = tf.compat.v1.random_normal((batch_size, latent_dim), mean=0.0, stddev=1.0, name='epsilon')

        with tf.name_scope('construct_Z'):
            # Sample epsilon from the Gaussian distribution. #
            # std_encoder: N x Z_dim
            std_encoder = tf.exp(logvar_encoder / 2)
            # z: N x Z_dim
            z = tf.add(mu_encoder, tf.multiply(std_encoder, epsilon), name='Z')

    # ============================== P(X|Z) - Decoder NN ============================== #

    # The encoder is a neural network with 2 hidden layers.
    with tf.name_scope('decoder'):
        with tf.name_scope('Phis'):
            # ph1: M2 x Z_dim
            ph1 = initialize_weight_variable([hidden_decoder_dim, latent_dim], name='ph1')
            # bias_phi1: 1 x M2
            bias_phi1 = initialize_bias_variable([hidden_decoder_dim], name='bias_phi1')

            # phi2: D x M2
            phi2 = initialize_weight_variable([input_dim, hidden_decoder_dim],
                                                                           name='phi2')
            # bias_phi2: 1 x D
            bias_phi2 = initialize_bias_variable([input_dim], name='bias_phi2')

        with tf.name_scope('hidden_layer1'):
            # Hidden layer 1 activation function of the decoder
            # hidden_layer_decoder: N x M2
            # RELU
            hidden_layer_decoder = tf.nn.relu(tf.nn.bias_add(tf.matmul(z, tf.transpose(ph1)), bias_phi1))
            # SIGMOID
            #hidden_layer_decoder = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(z, tf.transpose(ph1)), bias_phi1))

        with tf.name_scope('hidden_layer2'):
            # x_hat: N x D
            x_hat = tf.nn.bias_add(tf.matmul(hidden_layer_decoder, tf.transpose(phi2)), bias_phi2)

    with tf.name_scope('reconstructed_data'):
        # X_recon_samples: N x D, reconstructed data
        x_recon_samples = tf.nn.sigmoid(x_hat, name='X_recon_samples')

        # reshape images to the appropriate format and write to summary
        if input_dim == 784:  # for MNIST or OMNIGLOT datasets
            tf.compat.v1.summary.image('image_data', tf.reshape(x_recon_samples, shape=[-1, 28, 28, 1]))
        elif input_dim == 1024:  # for CIFAR-10 Grayscale dataset
            tf.compat.v1.summary.image('image_data', tf.reshape(x_recon_samples, shape=[-1, 32, 32, 1]))
        elif input_dim == 3072:  # for CIFAR-10 RGB dataset
            tf.compat.v1.summary.image('image_data', tf.reshape(x_recon_samples, shape=[-1, 32, 32, 3]))
        elif input_dim == 10304:  # for 'Faces' dataset
            tf.compat.v1.summary.image(
                'image_data',
                tf.transpose(tf.reshape(x_recon_samples, shape=[-1, 92, 112, 1]), perm=[0, 2, 1, 3])
            )

    # ============================== TRAINING ============================== #

    with tf.name_scope('ELBO'):
        with tf.name_scope('reconstruction_cost'):
            # log P(X)
            # reconstruction_cost: N x 1
            reconstruction_cost = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x),
                axis=1
            )

        with tf.name_scope('KL_divergence'):
            # KLD: N x 1
            KLD = 0.5 * tf.reduce_sum(
                1 + logvar_encoder - tf.square(mu_encoder) - tf.exp(logvar_encoder),
                axis=1
            )

        elbo = tf.reduce_mean(reconstruction_cost - KLD, name='lower_bound')

        loss_summ = tf.compat.v1.summary.scalar('ELBO', elbo)

        thetas = [theta1, bias_theta1, theta_mu, bias_theta_mu, theta_logvar, bias_theta_logvar]
        phis = [ph1, bias_phi1, phi2, bias_phi2]
        var_list = thetas + phis

        # Adam Optimizer (WORKS BEST!) #
        grads_and_vars = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).compute_gradients(
            loss=elbo,
            var_list=var_list
        )
        apply_updates = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).apply_gradients(
            grads_and_vars=grads_and_vars)

        # Gradient Descent Optimizer #
        '''
        grads_and_vars = tf.train.GradientDescentOptimizer(learning_rate=lr). \
                         compute_gradients(loss=elbo, var_list=thetas.extend(phis))
        apply_updates = tf.train.GradientDescentOptimizer(learning_rate=lr). \
                        apply_gradients(grads_and_vars=grads_and_vars)
        '''

    # add op for merging summary
    summary_op = tf.compat.v1.summary.merge_all()

    # add Saver ops
    saver = tf.compat.v1.train.Saver()

    return x, loss_summ, apply_updates, summary_op, saver, elbo, x_recon_samples

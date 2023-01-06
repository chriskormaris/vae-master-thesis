# Based on the Vanilla VAE with PyTorch from this url:
# https://github.com/wiseodd/generative-models

import numpy as np
import torch
import torch.nn.functional as nn
import torch.optim as optim
from torch.autograd import Variable


#####


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


def initialize_weights(X_dim, hidden_encoder_dim, hidden_decoder_dim, Z_dim, lr=0.01):
    # ============================== Encoder Parameters: Thetas ============================== #

    # M1 x D
    theta1 = xavier_init(size=[hidden_encoder_dim, X_dim])
    # 1 x M1
    bias_theta1 = Variable(torch.zeros(hidden_encoder_dim), requires_grad=True)

    # Z_dim x M1
    theta_mu = xavier_init(size=[Z_dim, hidden_encoder_dim])
    # 1 x Z_dim
    bias_theta_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

    # Z_dim x M1
    theta_logvar = xavier_init(size=[Z_dim, hidden_encoder_dim])
    # 1 x Z_dim
    bias_theta_logvar = Variable(torch.zeros(Z_dim), requires_grad=True)

    # ============================== Decoder Parameters: Phis ============================== #

    # M2 x Z_dim
    phi1 = xavier_init(size=[hidden_decoder_dim, Z_dim])
    # 1 x M2
    bias_phi1 = Variable(torch.zeros(hidden_decoder_dim), requires_grad=True)

    # D x M2
    phi2 = xavier_init(size=[X_dim, hidden_decoder_dim])
    # 1 x D
    bias_phi2 = Variable(torch.zeros(X_dim), requires_grad=True)

    # ============================== TRAINING ==============================

    thetas = [theta1, bias_theta1,
              theta_mu, bias_theta_mu,
              theta_logvar, bias_theta_logvar]
    phis = [phi1, bias_phi1,
            phi2, bias_phi2]
    params = thetas + phis

    solver = optim.Adam(params, lr=lr)

    return params, solver


# @: denotes matrix multiplication
def train(x, mb_size, Z_dim, params, solver):
    x = Variable(torch.from_numpy(x).float())

    theta1 = params[0]  # M1 x D
    bias_theta1 = params[1]  # 1 x M1
    theta_mu = params[2]  # Z_dim x M1
    bias_theta_mu = params[3]  # 1 x Z_dim
    theta_logvar = params[4]  # Z_dim x M1
    bias_theta_logvar = params[5]  # 1 x Z_dim
    phi1 = params[6]  # M2 x Z_dim
    bias_phi1 = params[7]  # 1 x M2
    phi2 = params[8]  # D x M2
    bias_phi2 = params[9]  # 1 x D

    # ============================== Q(Z|X) = Q(Z) - Encoder NN ============================== #

    # hidden_layer_encoder = nn.relu(x @ torch.transpose(theta1, 0, 1) + bias_theta1.repeat(mb_size, 1))
    hidden_layer_encoder = nn.relu(torch.mm(x, torch.transpose(theta1, 0, 1)) +
                                   bias_theta1.repeat(mb_size, 1))
    # mu_encoder = hidden_layer_encoder @ torch.transpose(theta_mu, 0, 1) + \
    #              bias_theta_mu.repeat(hidden_layer_encoder.size(0), 1)
    mu_encoder = torch.mm(hidden_layer_encoder, torch.transpose(theta_mu, 0, 1)) + \
                 bias_theta_mu.repeat(hidden_layer_encoder.size(0), 1)
    # logvar_encoder = hidden_layer_encoder @ torch.transpose(theta_logvar, 0, 1) + \
    #                  bias_theta_logvar.repeat(hidden_layer_encoder.size(0), 1)
    logvar_encoder = torch.mm(hidden_layer_encoder, torch.transpose(theta_logvar, 0, 1)) + \
                     bias_theta_logvar.repeat(hidden_layer_encoder.size(0), 1)

    # Sample epsilon from the Gaussian distribution. #
    epsilon = Variable(torch.randn(mb_size, Z_dim))
    # std_encoder: N x Z_dim
    std_encoder = torch.exp(logvar_encoder / 2)
    # Sample the latent variables Z. #
    z = mu_encoder + std_encoder * epsilon

    # ============================== P(X|Z) - Decoder NN ============================== #

    # hidden_layer_decoder = nn.relu(z @ torch.transpose(phi1, 0, 1) + bias_phi1.repeat(z.size(0), 1))
    hidden_layer_decoder = nn.relu(torch.mm(z, torch.transpose(phi1, 0, 1)) + bias_phi1.repeat(z.size(0), 1))
    # x_hat = hidden_layer_decoder @ torch.transpose(phi2, 0, 1) + bias_phi2.repeat(hidden_layer_decoder.size(0), 1)
    x_hat = torch.mm(hidden_layer_decoder, torch.transpose(phi2, 0, 1)) + \
            bias_phi2.repeat(hidden_layer_decoder.size(0), 1)
    x_recon_samples = torch.sigmoid(x_hat)

    # Loss #
    recon_loss = nn.binary_cross_entropy_with_logits(x_recon_samples, x, reduction='sum') / mb_size
    kl_loss = torch.mean(0.5 * torch.sum(1 + logvar_encoder - mu_encoder ** 2 - torch.exp(logvar_encoder), 1))
    elbo_loss = recon_loss - kl_loss

    # Backward #
    elbo_loss.backward()

    # Update #
    solver.step()

    # Housekeeping #
    for p in params:
        p.grad.data.zero_()

    # convert to numpy data type and return
    x_recon_samples = x_recon_samples.data.numpy()
    elbo_loss = elbo_loss.data.numpy()

    return x_recon_samples, elbo_loss

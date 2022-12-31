import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

__author__ = 'c.kormaris'


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

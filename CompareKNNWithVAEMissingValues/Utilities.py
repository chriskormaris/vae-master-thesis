import numpy as np

#####


def non_missing_percentage(X, missing_value=0.5):
    return np.where(X != missing_value)[0].size / np.size(X) * 100


def non_zero_percentage(X):
    return np.count_nonzero(X) / np.size(X) * 100


# root mean squared error
def rmse(X, X_recon):
    return np.sqrt(np.mean(np.square(X - X_recon)))


# mean absolute error
def mae(X, X_recon):
    return np.mean(np.abs(X - X_recon))

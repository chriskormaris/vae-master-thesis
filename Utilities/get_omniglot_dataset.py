import scipy.io as sio
import numpy as np


# The dataset contains 50 alphabets. Each alphabet has 60 characters.
# The alphabet number 20 (the 21st) is the Greek one
# The alphabet number 31 (the 32nd) is the English one
def get_omniglot_dataset(filepath, train_or_test, alphabet, binarize=False):
    all = sio.loadmat(filepath)

    X = None
    alphabet_labels_one_hot = None
    y = None

    if train_or_test.lower() == 'train':
        X = all.get('data').T  # X: 24345 x 784
        alphabet_labels_one_hot = all.get('target').T  # X: 24345 x 50
        y = np.squeeze(all.get('targetchar'))  # X: 24345 x 1, takes values in the range [1, 26]
    elif train_or_test.lower() == 'test':
        X = all.get('testdata').T  # X: 8070 x 784
        alphabet_labels_one_hot = all.get('testtarget').T  # X: 8070 x 50
        y = np.squeeze(all.get('testtargetchar'))  # X: 8070 x 1, takes values in the range [1, 26]

    alphabet_labels = np.argmax(alphabet_labels_one_hot, axis=1)  # X: 24345 x 1
    X = np.squeeze(X[np.where(alphabet_labels == alphabet), :])

    # rotate images correctly
    X = X.reshape((-1, 28, 28))
    X = np.transpose(X, axes=(0, 2, 1))
    X = X.reshape((-1, 784))

    if binarize:
        X = binarize_omniglot_dataset(X)

    y = y[np.where(alphabet_labels == alphabet)]

    return X, y


# convert omniglot dataset values to 0s and 1s
def binarize_omniglot_dataset(X):
    return np.round(X + 0.3)

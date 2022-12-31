import numpy as np

__author__ = 'c.kormaris'

# ignore errors
np.seterr(all='ignore')


##########


# Customized Euclidean distance function.
def sqdist(X_train, X_test_instance, missing_value):
    Ntrain = X_train.shape[0]
    D = X_train.shape[1]

    # Make the train data missing, where the current
    # test instance has missing values.
    # Thus, the difference X_train_common - X_test_common,
    # in the indices where the test instance values
    # are missing, will be equal to missing_value.
    X_train_common = np.array(X_train)
    s = np.where(X_test_instance == missing_value)
    X_train_common[:, s] = missing_value

    # Repeat the test instance Ntrain times and
    # make the test data values, equal to the mean
    # of all train examples, where the current
    # train instance has missing values.
    X_test_common = np.zeros((Ntrain, D))
    mean_values = np.mean(X_train, axis=0)  # 1 x D array
    for k in range(Ntrain):
        X_test_common[k, :] = X_test_instance
        s = np.where(X_train[k, :] == missing_value)
        if len(s[0]) != 0:
            X_test_common[k, s] = mean_values[s]

    dist = np.sqrt(np.sum(np.square(X_train_common - X_test_common), axis=1)).T
    return dist


def kNNMatrixCompletion(X_train, X_test, K, missing_value, use_softmax_weights=True, binarize=False):
    Ntest = X_test.shape[0]

    X_test_predicted = np.array(X_test)
    for i in range(Ntest):
        print('data_%i' % i)
        X_test_i = np.squeeze(np.array(X_test[i, :]))

        distances = sqdist(X_train, X_test_i, missing_value)

        # sort the distances in ascending order
        # and store the indices in a variable
        closest_data_indices = (np.argsort(distances, axis=0)).tolist()

        # select the top k indices
        closest_k_data_indices = closest_data_indices[:K]
        closest_k_data = np.squeeze(X_train[closest_k_data_indices, :])

        closest_k_data_distances = distances[closest_k_data_indices]

        if use_softmax_weights:
            # distribute weights, in the following manner:
            # wk = e^(-k) / sum_{i=1}^{K} e^(-i)
            # such that: w1 + w2 + ... wK = 1
            a = 1
            weights = softmax(-a * closest_k_data_distances)
        else:
            # ALTERNATIVE: all weights equal to 1 / K
            weights = np.ones((K, 1)) / float(K)

        weights = np.reshape(weights, newshape=(K, 1))
        closest_k_data = closest_k_data * np.repeat(weights, closest_k_data.shape[1], axis=1)

        # sum across all rows for each column, returns a column vector
        predicted_pixels = np.sum(closest_k_data, axis=0).T

        X_test_predicted[i, np.where(X_test_i == missing_value)] = \
                predicted_pixels[np.where(X_test_i == missing_value)].T

    if binarize:
        X_test_predicted = np.round(X_test_predicted + 0.3)

    return X_test_predicted


def softmax(x):
    return np.divide(np.exp(x), np.sum(np.exp(x), axis=0))

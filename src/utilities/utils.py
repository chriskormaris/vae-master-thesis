import numpy as np


#####


def reduce_data(X, num_examples, reduced_num_examples, y=None, t=None, num_classes=10):
    starting_index = int(num_examples / num_classes)
    offset = int(reduced_num_examples / num_classes)

    X = np.concatenate((X[:offset, :],
                        X[starting_index: starting_index + offset, :],
                        X[starting_index * 2: starting_index * 2 + offset, :],
                        X[starting_index * 3: starting_index * 3 + offset, :],
                        X[starting_index * 4: starting_index * 4 + offset, :],
                        X[starting_index * 5: starting_index * 5 + offset, :],
                        X[starting_index * 6: starting_index * 6 + offset, :],
                        X[starting_index * 7: starting_index * 7 + offset, :],
                        X[starting_index * 8: starting_index * 8 + offset, :],
                        X[starting_index * 9: starting_index * 9 + offset, :]))

    if y is not None:
        y = np.concatenate((y[:offset],
                            y[starting_index: starting_index + offset],
                            y[starting_index * 2: starting_index * 2 + offset],
                            y[starting_index * 3: starting_index * 3 + offset],
                            y[starting_index * 4: starting_index * 4 + offset],
                            y[starting_index * 5: starting_index * 5 + offset],
                            y[starting_index * 6: starting_index * 6 + offset],
                            y[starting_index * 7: starting_index * 7 + offset],
                            y[starting_index * 8: starting_index * 8 + offset],
                            y[starting_index * 9: starting_index * 9 + offset]))

    if t is not None:
        t = np.concatenate((t[:offset, :],
                            t[starting_index: starting_index + offset, :],
                            t[starting_index * 2: starting_index * 2 + offset, :],
                            t[starting_index * 3: starting_index * 3 + offset, :],
                            t[starting_index * 4: starting_index * 4 + offset, :],
                            t[starting_index * 5: starting_index * 5 + offset, :],
                            t[starting_index * 6: starting_index * 6 + offset, :],
                            t[starting_index * 7: starting_index * 7 + offset, :],
                            t[starting_index * 8: starting_index * 8 + offset, :],
                            t[starting_index * 9: starting_index * 9 + offset, :]))

    return X, y, t


def construct_missing_data(X, y=None, missing_value=0.5, structured_or_random='structured'):
    X_missing = np.array(X)

    # shuffle the data
    s = np.random.permutation(X.shape[0])
    # ALTERNATIVE
    # s = np.arange(X.shape[0])
    # np.random.shuffle(s)

    X = X[s, :]
    X_missing = X_missing[s, :]

    if y is not None:
        y = y[s]

    if structured_or_random == 'structured':
        # Construct the missing data, while keeping 1/5 of the data intact.
        # for MNIST, Binarized MNIST, OMNIGLOT and grayscaled CIFAR-10 datasets
        if X.shape[1] == 784 or X.shape[1] == 1024:

            # reshape data to 3 dimensions
            X_missing = np.reshape(X_missing, newshape=(-1, int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

            # make the lower half pixels missing
            X_missing[:int(X_missing.shape[0] / 5), int(X_missing.shape[1] / 2):, :] = missing_value

            # make the upper half pixels missing
            X_missing[
            int(X_missing.shape[0] / 5):int(X_missing.shape[0] * 2 / 5),
            :int(X_missing.shape[1] / 2),
            :] = missing_value

            # make the left half pixels missing
            X_missing[
            int(X_missing.shape[0] * 2 / 5):int(X_missing.shape[0] * 3 / 5),
            :,
            :int(X_missing.shape[2] / 2)] = missing_value

            # make the right half pixels missing
            X_missing[
            int(X_missing.shape[0] * 3 / 5):int(X_missing.shape[0] * 4 / 5),
            :,
            int(X_missing.shape[2] / 2):] = missing_value

            # reshape data back to 2 dimensions
            X_missing = np.reshape(X_missing, newshape=(-1, int(np.square(X_missing.shape[1]))))

        elif X.shape[1] == 3072:  # for RGB CIFAR-10 dataset
            # reshape data to 4 dimensions to make it easier
            X_missing = np.reshape(X_missing, newshape=(-1, 32, 32, 3))

            # make the lower half pixels missing
            X_missing[:int(X_missing.shape[0] / 5), int(X_missing.shape[1] / 3):, :, :] = missing_value

            # make the upper half pixels missing
            X_missing[
            int(X_missing.shape[0] / 5):int(X_missing.shape[0] * 2 / 5),
            :int(X_missing.shape[1] / 2),
            :,
            :] = missing_value

            # make the left half pixels missing
            X_missing[
            int(X_missing.shape[0] * 2 / 5):int(X_missing.shape[0] * 3 / 5),
            :,
            :int(X_missing.shape[2] / 2),
            :] = missing_value

            # make the right half pixels missing
            X_missing[
            int(X_missing.shape[0] * 3 / 5):int(X_missing.shape[0] * 4 / 5),
            :,
            int(X_missing.shape[2] / 2):,
            :] = missing_value

            # reshape data back to 2 dimensions
            X_missing = np.reshape(X_missing, newshape=(-1, 3072))

        elif X.shape[1] == 10304:  # for ORL Faces dataset
            # reshape data to 3 dimensions
            X_missing = np.reshape(X_missing, newshape=(-1, 92, 112))
            X_missing = np.transpose(X_missing, axes=[0, 2, 1])

            # make the lower half pixels missing
            X_missing[:int(X_missing.shape[0] / 5), int(X_missing.shape[1] / 3):, :] = missing_value

            # make the upper half pixels missing
            X_missing[
            int(X_missing.shape[0] / 5):int(X_missing.shape[0] * 2 / 5),
            :int(X_missing.shape[1] / 2),
            :] = missing_value

            # make the left half pixels missing
            X_missing[
            int(X_missing.shape[0] * 2 / 5):int(X_missing.shape[0] * 3 / 5),
            :,
            :int(X_missing.shape[2] / 2)] = missing_value

            # make the right half pixels missing
            X_missing[
            int(X_missing.shape[0] * 3 / 5):int(X_missing.shape[0] * 4 / 5),
            :,
            int(X_missing.shape[2] / 2):] = missing_value

            # reshape data back to 2 dimensions
            X_missing = np.transpose(X_missing, axes=[0, 2, 1])
            X_missing = np.reshape(X_missing, newshape=(-1, 10304))
        elif X.shape[1] == 32256:  # for YALE Faces dataset
            # reshape data to 3 dimensions
            X_missing = np.reshape(X_missing, newshape=(-1, 168, 192))
            X_missing = np.transpose(X_missing, axes=[0, 2, 1])

            # make the lower half pixels missing
            X_missing[:int(X_missing.shape[0] / 5), int(X_missing.shape[1] / 3):, :] = missing_value

            # make the upper half pixels missing
            X_missing[
            int(X_missing.shape[0] / 5):int(X_missing.shape[0] * 2 / 5),
            :int(X_missing.shape[1] / 2),
            :] = missing_value

            # make the left half pixels missing
            X_missing[
            int(X_missing.shape[0] * 2 / 5):int(X_missing.shape[0] * 3 / 5),
            :,
            :int(X_missing.shape[2] / 2)] = missing_value

            # make the right half pixels missing
            X_missing[
            int(X_missing.shape[0] * 3 / 5):int(X_missing.shape[0] * 4 / 5),
            :,
            int(X_missing.shape[2] / 2):] = missing_value

            # reshape data back to 2 dimensions
            X_missing = np.transpose(X_missing, axes=[0, 2, 1])
            X_missing = np.reshape(X_missing, newshape=(-1, 32256))

    elif structured_or_random == 'random':
        # make half the pixels missing at random
        for i in range(0, int(X_missing.shape[0] * 4 / 5)):
            s = np.random.permutation(X_missing.shape[1])
            # ALTERNATIVE
            # s = np.arange(X_missing.shape[1])
            # np.random.shuffle(s)
            s = s[int(s.shape[0] / 2):]
            X_missing[i, s] = missing_value

    # reshuffle the data
    s = np.random.permutation(X.shape[0])
    # ALTERNATIVE
    # s = np.arange(X.shape[0])
    # np.random.shuffle(s)

    X = X[s, :]
    X_missing = X_missing[s, :]

    if y is not None:
        y = y[s]

    return X_missing, X, y


def get_non_missing_percentage(X, missing_value=0.5):
    return np.where(X != missing_value)[0].size / np.size(X) * 100


def get_non_zero_percentage(X):
    return np.count_nonzero(X) / np.size(X) * 100


# root mean squared error
def rmse(X, X_recon):
    return np.sqrt(np.mean(np.square(X - X_recon)))


# mean absolute error
def mae(X, X_recon):
    return np.mean(np.abs(X - X_recon))

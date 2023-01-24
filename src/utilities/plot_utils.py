import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# Shows images in a 10 x 10 grid.
def plot_images(X, y, categories=None, n=10, title='', grayscale=True, show_plot=False):
    input_dim = np.prod(X.shape[1:])
    if categories is None:
        categories = list(range(10))
    fig = plt.figure(figsize=(10, 10))
    for c, category in enumerate(categories):
        if c * n + 1 < n * len(categories):
            i = 0
            # plot the first n data of each category
            for col in range(n):
                while i < len(y) and y[i] != category:
                    i = i + 1
                if i >= len(y):
                    break
                ax = plt.subplot(10, 10, col + c * n + 1)
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')

                if input_dim == 784:
                    image = X[i].reshape(28, 28)
                elif input_dim == 1024:
                    image = X[i].reshape(32, 32)
                elif input_dim == 3072:
                    image = X[i].reshape(32, 32, 3)
                elif input_dim == 10304:
                    image = X[i].reshape(112, 92)
                elif input_dim == 32256:
                    image = X[i].reshape(192, 168)
                else:
                    image = X[i]

                if grayscale:
                    # grayscale
                    plt.imshow(image, cmap='gray')
                    plt.gray()
                else:
                    plt.imshow(image)

                i = i + 1

    fig.canvas.manager.set_window_title(title)
    if show_plot:
        plt.show()

    return fig


# the samples must be of shape: 16 x input_dim
def plot_patches(samples, grayscale=True):
    input_dim = np.prod(samples.shape[1:])

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        if input_dim == 784:
            image = sample.reshape(28, 28)
        elif input_dim == 1024:
            image = sample.reshape(32, 32)
        elif input_dim == 3072:
            image = sample.reshape(32, 32, 3)
        elif input_dim == 10304:
            image = sample.reshape(112, 92)
        elif input_dim == 32256:
            image = sample.reshape(192, 168)
        else:
            image = sample

        if grayscale:
            plt.imshow(image, cmap='Greys_r')
        else:
            plt.imshow(image)

    return fig


def plot_movielens_data(data, plot_show=False):
    x = list(range(data.shape[1]))  # movies
    y = np.mean(data, axis=0)
    # plt.hist(x, y)
    plt.bar(x, y)
    if plot_show:
        plt.show()
    fig = plt.gcf()
    return fig

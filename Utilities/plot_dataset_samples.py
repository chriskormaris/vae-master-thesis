import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# used for MNIST or OMNIGLOT dataset
def plot_mnist_or_omniglot_data(X, y, categories=list(range(10)), n=10, title='', grayscale=True, show_plot=False):
    # show n samples for each class
    fig = plt.figure(figsize=(n, len(categories)))
    for c, category in enumerate(categories):
        if int(n * c+1) < int(n*len(categories)):
            i = 0
            # plot the first n data of each category
            for col in range(len(categories)):
                while y[i] != category:
                    i = i + 1
                ax = plt.subplot(10, 10, col + c * 10 + 1)
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')

                image = X[i].reshape(28, 28)

                if grayscale:
                    # grayscale
                    plt.imshow(image, cmap='gray')
                    plt.gray()
                else:
                    plt.imshow(image)

                i = i + 1
    # fig.canvas.set_window_title(title)
    if show_plot:
        plt.show()
    return fig


def plot_cifar10_data(X, y, categories=list(range(10)), n=10, title='', grayscale=False, show_plot=False):
    # show n samples for each class
    fig = plt.figure(figsize=(n, len(categories)))
    for c, category in enumerate(categories):
        if int(n * c+1) < int(n * len(categories)):
            i = 0
            # plot the first n data of each category
            for col in range(len(categories)):
                while y[i] != category:
                    i = i + 1
                ax = plt.subplot(10, 10, col + c * n + 1)
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')

                if grayscale:
                    # 1024 = 32 x 32
                    image = X[i].reshape(32, 32)
                    plt.imshow(image, cmap='Greys_r')
                    plt.gray()
                else:
                    # 3072 = 32 x 32 x 3
                    image = X[i].reshape(32, 32, 3)
                    plt.imshow(image)

                i = i + 1
    # fig.canvas.set_window_title(title)
    if show_plot:
        plt.show()
    return fig


def plot_orl_faces(X, y, categories=list(range(1, 11)), n=10, title='', grayscale=True, show_plot=False):
    # show n samples for each class
    fig = plt.figure(figsize=(n, len(categories)))
    for c, category in enumerate(categories):
        if int(n * c + 1) < int(n * len(categories)):
            i = 0
            # plot the first n data of each category
            for col in range(len(categories)):
                while y[i] != category:
                    i = i + 1
                ax = plt.subplot(10, 10, col + c * 10 + 1)
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')

                image = X[i].reshape(112, 92)

                if grayscale:
                    # grayscale
                    plt.imshow(image, cmap='gray')
                    plt.gray()
                else:
                    plt.imshow(image)

                i = i + 1

    # fig.canvas.set_window_title(title)
    if show_plot:
        plt.show()
    return fig


def plot_yale_faces(X, y, categories=list(range(1, 11)), n=10, title='', grayscale=True, show_plot=False):
    # show n samples for each class
    fig = plt.figure(figsize=(n, len(categories)))
    for c, category in enumerate(categories):
        if int(n * c + 1) < int(n * len(categories)):
            i = 0
            # plot the first n data of each category
            for col in range(n):
                while y[i] != category:
                    i = i + 1
                ax = plt.subplot(10, 10, col + c * 10 + 1)
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')

                image = X[i].reshape(192, 168)

                if grayscale:
                    # grayscale
                    plt.imshow(image, cmap='gray')
                    plt.gray()
                else:
                    plt.imshow(image)

                i = i + 1

    # fig.canvas.set_window_title(title)
    if show_plot:
        plt.show()
    return fig


# the samples must be of shape: 16 x 784
def plot_mnist_or_omniglot_patches(samples, grayscale=True):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        image = sample.reshape(28, 28)
        if grayscale:
            plt.imshow(image, cmap='Greys_r')
        else:
            plt.imshow(image)

    return fig


# the samples must be of shape: 16 x 3072 or 16 x 1024
def plot_cifar10_patches(samples, grayscale=True):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        if grayscale:
            # 1024 = 32 x 32
            image = sample.reshape(32, 32)
            plt.imshow(image, cmap='Greys_r')
            plt.gray()
        else:
            # 3072 = 32 x 32 x 3
            image = sample.reshape(32, 32, 3)
            plt.imshow(image)

    return fig


# the samples must be of shape: 16 x 10304
def plot_orl_faces_patches(samples, grayscale=True):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        image = sample.reshape(92, 112).T
        if grayscale:
            plt.imshow(image, cmap='Greys_r')
        else:
            plt.imshow(image)

    return fig


# the samples must be of shape: 16 x 32256
def plot_yale_faces_patches(samples, grayscale=True):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        image = sample.reshape(192, 168)
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

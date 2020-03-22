# Variational Autoencoder Thesis #

Programming Language: Python 3

Made by Christos Kormaris

Supervisor Professor: Michalis Titsias

This repository was created as a part of my Master Thesis, while studying in Athens University of Economics & Business (AUEB).

## Description ##

This is an implementation of variational autoencoders, using various datasets.
The datasets included are: MNIST, Binarized MNIST, CIFAR-10 and OMNIGLOT.
There is also an implementation on the "Database of Faces" dataset, but the data are very few to get good results.
The autoencoder creates new artificial data and to be precise images, from the given original image data.

You can download all the datasets here:
[https://www.dropbox.com/sh/ucvad0dkcbxuyho/AAAjjrRPYiGLLPc_VKru4-Uva?dl=0](https://www.dropbox.com/sh/ucvad0dkcbxuyho/AAAjjrRPYiGLLPc_VKru4-Uva?dl=0)


## How to set up and run the VAEs GUI

A graphical user interface (GUI) has been implemented for the project of this thesis, using Python and the Tkinter library.

First, browse to the directory "code" and install the required Python dependencies, by typing:
```python
pip install −r depencencies .txt
```

Then, extract the "EXE.zip" archive in the same directory.
Open the EXE folder, find the file vaes_gui.exe and execute it.

Finally, download all the datasets from the URL in the file "datasets_urls.txt" and move them to the newly created "dist" folder.
Inside, there should be a folder with the name "vaes_gui", which contains the executable file "vaes_gui.exe".

The GUI will make it all easier for you now.

### Screenshot

![vaes_001](/screenshots/vaes_gui/vaes_001.png)


### MNIST Dataset ###

Extract the compressed file "MNIST_dataset.zip".
A folder named "MNIST_dataset" should appear, which contains the files of the MNIST dataset, along with their labels.

### Binarized MNIST Dataset ###

Extract the compressed file "Binarized_MNIST_dataset.zip".
A folder named "Binarized_MNIST_dataset" should appear, which contains the TRAIN, TEST and VALIDATION files of the Binarized MNIST dataset, along with labels only for the TEST data.


**Note: ** Uncomment the command *language = 'english'* or *language = 'greek'* to use the english or greek alphabet respectively.

### CIFAR-10 Dataset ###

Extract the compressed file "CIFAR_daset.zip".
A folder named "CIFAR_dataset" should appear, which contains the TRAIN and TEST files of the CIFAR-10 and CIFAR-100 dataset, along with their labels. The CIFAR-10 dataset contains data from 10 classes, while the CIFAR-100 dataset contains data from 100 classes.


**Note: ** Uncomment the command *RGBOrGrayscale = 'RGB'* or *RGBOrGrayscale = 'grayscale'* to use the colored or grayscaled images respectively. In the TensorFlow implementation, the results are good only if the images are grayscaled!

### OMNIGLOT Dataset ###

Extract the compressed file "OMNIGLOT_daset.zip".
A folder named "OMNIGLOT_dataset" should appear, which contains the TRAIN and TEST files of the OMNIGLOT dataset, from 50 different alphabets, along with their labels.
Two alphabets are used, the Greek and the English.
The Greek alphabet has 24 characters, which means 24 are the classes.
The Greek alphabet has 26 characters, which means 26 are the classes.
The classes are not important for the algorithm, but they are used for plotting purposes.


# K-NN Missing Values algorithms #

These are implementations of K-NN Missing Values algorithms on various datasets with missing values.
The datasets included are: MNIST, Binarized MNIST, CIFAR-10 and OMNIGLOT.
I've implemented an algorithm that uses K-NN for regression, i.e. it predicts the missing pixel values, based on the corresponding pixels of the top K nearest neighbors.

### MNIST Dataset ###

The results of the algorithm will be new images of digits, with their missing halves predicted.

### Binarized MNIST Dataset ###

The results of the algorithm will be new images of binarized digits, with their missing halves predicted.

### CIFAR-10 Dataset ###

The results of the algorithm will be new images of cats and dogs, with their missing halves predicted.

### OMNIGLOT Dataset ###

The results of the algorithm will be new images of alphabet characters, with their missing halves predicted.

### The Database of Faces Dataset ###

[//]: # (### How to set up and run the K-NN Missing Values algorithm on the 'The Database of Faces' Dataset ###)
[//]: # (Extract the compressed file "TheDatabaseOfFaces_dataset.zip".)
[//]: # (A folder named "TheDatabaseOfFaces_dataset" should appear, which contains a file with 'The Database of Faces' dataset.)
[//]: # (In the dataset, there are 400 face images in total, from 40 different persons and 10 images from each person, 40 * 10 = 400.)
[//]: # (The results of the algorithm will be new images of the faces, with their missing halves predicted.)


# VAE Missing Values algorithms #

There are also implementations of Variational Autoencoder Missing Values algorithms, on various datasets with missing values.
There are two different implementations of VAEs included: 1) using TensorFlow and 2) using PyTorch
The datasets included are: MNIST, Binarized MNIST, CIFAR-10 and OMNIGLOT.
The algorithm uses a Variational Autoencoder to predict only the missing pixel values, based on the training data.

**Note: ** In some datasets, e.g. in the CIFAR-10 dataset, the results are good only if the images are grayscaled!

### Tensorboard ###

To open and examine a visualization of the autoencoder Missing Values,
change your working directory to the location "VAEsMissingValuesInTensorFlow"
and run the following commands from the command prompt on Windows, or the terminal on Linux:

```shell
tensorboard --logdir=./logs/mnist
```

```shell
tensorboard --logdir=./logs/binarized_mnist
```

```shell
tensorboard --logdir=./logs/cifar10_rgb
```

```shell
tensorboard --logdir=./logs/cifar10_grayscale
```

```shell
tensorboard --logdir=./logs/omniglot_english
```

```shell
tensorboard --logdir=./logs/omniglot_greek
```

Then, open your browser ang navigate to -> http://localhost:6006

### Small Tensorboard screenshot
![vae_tensorboard_graph_with_reconstructed_data](/screenshots/tensorboard/vae_tensorboard_graph_with_reconstructed_data.png)

### Large Tensorboard screenshot
![vae_tensorboard_graph_with_reconstructed_data_large](/screenshots/tensorboard/vae_tensorboard_graph_with_reconstructed_data_large.png)

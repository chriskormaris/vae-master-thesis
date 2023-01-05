# Variational Autoencoders & Applications Master Thesis #

Programming Language: Python 3

GUI toolkit: tkinter

Made by Christos Kormaris, April-May 2018

Supervisor Professor: Michalis Titsias

This repository was created for the purposes of my Master Thesis for the MSc in Computer Science at Athens University of Economics & Business (AUEB).

## ResearchGate links ##
You can find the `.pdf` files of my master thesis on the site of ResearchGate in two languages, English & Greek:
- English translation: https://www.researchgate.net/publication/337000568
- Greek translation: https://www.researchgate.net/publication/349465619


## Description ##

The variational autoencoder is a process that creates new artificial data, which are in many cases images, from the original data.
This repository contains implementations of variational autoencoders, on various datasets.
The datasets included are: MNIST, Binarized MNIST, CIFAR-10 and OMNIGLOT, ORL Face Database & Yale Faces.
On the `ORL Face Database` & `Yale Faces` datasets, the data are very few and the variational autoencoder implementations do not have good results.

You can download all the datasets here:
[https://www.dropbox.com/sh/ucvad0dkcbxuyho/AAAjjrRPYiGLLPc_VKru4-Uva?dl=0](https://www.dropbox.com/sh/ucvad0dkcbxuyho/AAAjjrRPYiGLLPc_VKru4-Uva?dl=0)


## How to run the VAEs GUI

A graphical user interface (GUI) has been implemented for the project of this thesis, using Python 3 and the Tkinter library.

First, install all requirements:
```shell
pip install -r requirements.txt
```
Then, run:
```shell
python vaes_gui.py
```

## How to create an executable for the GUI

1. First, install the `pyinstaller` dependency.
   ```shell
   pip install pyinstaller
   ```
2. Then, make an executable file using the settings from the file [vaes_gui.spec](vaes_gui.spec)
   ```shell
   pyinstaller vaes_gui.spec
   ```

Then, download all the datasets from the URL in the file `datasets_urls.md`, extract them and move them to the `dist` folder.
Now, browse to the directory `dist/vaes_gui`. Find the file `vaes_gui.exe` and run it.
The GUI will make it all easier for you!

### Screenshot

![vaes_001](/screenshots/vaes_gui/vaes_001.png)


### MNIST Dataset ###

Extract the compressed file `MNIST_dataset.zip`.
A folder named `MNIST_dataset` should appear, which contains the files of the MNIST dataset, along with their labels.

### Binarized MNIST Dataset ###

Extract the compressed file `Binarized_MNIST_dataset.zip`.
A folder named `Binarized_MNIST_dataset` should appear, which contains the TRAIN, TEST and VALIDATION files of the Binarized MNIST dataset, along with labels only for the TEST data.

### CIFAR-10 Dataset ###

Extract the compressed file `CIFAR_daset.zip`.
A folder named `CIFAR_dataset` should appear, which contains the TRAIN and TEST files of the CIFAR-10 and CIFAR-100 dataset, along with their labels. The CIFAR-10 dataset contains data from 10 classes, while the CIFAR-100 dataset contains data from 100 classes.

### OMNIGLOT Dataset ###

Extract the compressed file `OMNIGLOT_daset.zip`.
A folder named `OMNIGLOT_dataset` should appear, which contains the TRAIN and TEST files of the OMNIGLOT dataset, from 50 different alphabets, along with their labels.
Two alphabets are used, the Greek and the English.
The Greek alphabet has 24 characters, which means 24 are the classes.
The Greek alphabet has 26 characters, which means 26 are the classes.
The classes are not important for the algorithm, but they are used for plotting purposes.


# K-NN Missing Values algorithm #

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

### ORL Face Database Dataset ###

### How to set up and run the K-NN Missing Values algorithm on the ORL Face Database dataset ###
Extract the compressed file `ORL_Face_Dataset.zip`.
Create a folder named `ORL_Face_Dataset` and unzip there the contents of the zip file.
In the dataset, there are 400 face images in total, from 40 different persons and 10 images from each person, 40 * 10 = 400.
The results of the algorithm will be new images of the faces, with their missing halves predicted.


# VAE Missing Values algorithm #

There are also two different implementations of the Variational Autoencoder Missing Values algorithm of the VAEs included: 
1) using TensorFlow and 
2) using PyTorch
The datasets included are: MNIST, Binarized MNIST, CIFAR-10, OMNIGLOT, ORL Face Database, Yale Faces & the Movielens dataset.
The algorithm uses a Variational Autoencoder to predict only the missing pixel values, based on the training data.

**Note:** In some datasets, e.g. in the CIFAR-10 dataset, the results are good only if the images are grayscaled!

### Tensorboard ###

To open and examine a visualization of the autoencoders, change your working directory to the executable files folder `vaes_gui`
and run the following commands from the command prompt on Windows, or the terminal on Linux:

```shell
tensorboard --logdir=./tensorflow_logs/mnist_vae
```

```shell
tensorboard --logdir=./tensorflow_logs/binarized_mnist_vae
```

```shell
tensorboard --logdir=./tensorflow_logs/cifar10_rgb_vae
```

```shell
tensorboard --logdir=./tensorflow_logs/cifar10_grayscale_vae
```

```shell
tensorboard --logdir=./tensorflow_logs/omniglot_english_vae
```

```shell
tensorboard --logdir=./tensorflow_logs/omniglot_greek_vae
```

```shell
tensorboard --logdir=./tensorflow_logs/orl_faces_vae
```

```shell
tensorboard --logdir=./tensorflow_logs/yale_faces_vae
```

Then, open your browser ang navigate to -> http://localhost:6006

Similarly, you can open tensorboards for the implementation of VAE missing values algorithm in TensorFlow, by replacing the `_vae` postfix with `_vae_missing_values`.
In addition, the Keras implementation of the VAEs has its own logs, located in the folder `keras_logs`.

### Small Tensorboard screenshot
![vae_tensorboard_graph_with_reconstructed_data](/screenshots/tensorboard/vae_tensorboard_graph_with_reconstructed_data.png)

### Large Tensorboard screenshot
![vae_tensorboard_graph_with_reconstructed_data_large](/screenshots/tensorboard/vae_tensorboard_graph_with_reconstructed_data_large.png)

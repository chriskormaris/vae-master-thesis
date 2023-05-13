# Variational Autoencoders & Applications Master Thesis #

Programming Language: Python 3

GUI toolkit: tkinter

Made by Christos Kormaris, April-May 2018

Supervisor Professor: Michalis Titsias

This repository was created for the purposes of my Master Thesis for the MSc in Computer Science at Athens University of Economics & Business (AUEB).


## ResearchGate links ##
You can find the `.pdf` files of my master thesis on the site of ResearchGate in two languages, English & Greek.

* English translation: https://www.researchgate.net/publication/337000568
* Greek translation: https://www.researchgate.net/publication/349465619


## Abstract ##

A variational autoencoder is a method that can produce artificial data which will resemble a given dataset of real data.
For instance, if we want to produce new artificial images of cats, we can use a variational autoencoder algorithm to do so,
after training on a large dataset of images of cats.
The input dataset is unlabeled on the grounds that we are not interested in classifying the data to a specific class,
but we would rather be able to learn the most important features or similarities among the data.
Since the data are not labeled, the variational autoencoder is described as an unsupervised learning algorithm,
and it belongs in the area known as Reinforcement Learning.
As far as the example of cat images is concerned,
the algorithm can learn to detect that a cat should have two ears, a nose, whiskers, four legs, a tail and a diversity of colors.
The algorithm uses two neural networks, an encoder and a decoder, which are trained simultaneously.
A variational autoencoder should have good applications in cases where we would like to produce a bigger dataset,
for better training on various neural networks. Also, it runs dimensionality reduction on the initial data,
by compressing them into latent variables.
We run implementations of variational autoencoders on various datasets,
MNIST, Binarized MNIST, CIFAR-10, OMNIGLOT, YALE Faces, The Database of Faces, MovieLens,
written in Python 3 with three different libraries, TensorFlow, PyTorch and Keras and we present the results.
We introduce a simple missing values completion algorithm using K-NN collaborative filtering for making predictions (e.g. on missing pixels).
Finally, we make use of the variational autoencoders to run missing values completion algorithms and predict missing values on various datasets.
The K-NN algorithm did surprisingly well on the predictions, while the variational autoencoder completion system brought very satisfactory results.
A graphical user interface has also been implemented as well.

### Variational Autoencoder structure
<img src="images/autoencoder_structure.png" width="75%" height="75%">

**NOTE:**

You can download all the datasets from here:

[https://www.dropbox.com/sh/ucvad0dkcbxuyho/AAAjjrRPYiGLLPc_VKru4-Uva?dl=0](https://www.dropbox.com/sh/ucvad0dkcbxuyho/AAAjjrRPYiGLLPc_VKru4-Uva?dl=0)


## Extract datasets

Go one level up from the project directory and create the directory `DATASETS`.
Then, download all the datasets from the URL in the file `datasets_urls.md`, extract them and move them to the `DATASETS` folder.


## How to run the VAEs GUI

A graphical user interface (GUI) has been implemented for the project of this thesis, using Python 3 and the tkinter library.

Go to the project directory.

First, install all requirements:
```shell
pip install -r requirements.txt
```
Then, run:
```shell
python vaes_gui.py
```


### GUI Screenshots

#### Welcome screen
![vaes_001](/screenshots/vaes_gui/vaes_001.png)

#### Algorithm parameters selection screen
![vaes_001](/screenshots/vaes_gui/vaes_002.png)

#### About screen
![About_001.png](screenshots%2Fvaes_gui%2FAbout_001.png)

#### Datasets screen
![About_002.png](screenshots%2Fvaes_gui%2FAbout_002.png)

### Datasets

#### MNIST Dataset ####

Extract the compressed file `MNIST_dataset.zip`.
A folder named `MNIST_dataset` should appear, which contains the files of the MNIST dataset, along with their labels.

##### VAE in TensorFlow output images
|                                Original data                                |                               Epoch 1                               |                              Epoch 20                               |
|:---------------------------------------------------------------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------:|
| ![original_data](/output_images/vaes_in_tensorflow/mnist/original_data.png) | ![epoch_001](/output_images/vaes_in_tensorflow/mnist/epoch_001.png) | ![epoch_020](/output_images/vaes_in_tensorflow/mnist/epoch_020.png) |

##### VAE in Keras output images
![mnist.png](output_images%2Fvaes_in_keras%2Fmnist.png)

#### Binarized MNIST Dataset ####

Extract the compressed file `Binarized_MNIST_dataset.zip`.
A folder named `Binarized_MNIST_dataset` should appear, which contains the TRAIN, TEST and VALIDATION files of the Binarized MNIST dataset, along with labels only for the TEST data.

##### VAE in Keras output images
![binarized_mnist.png](output_images%2Fvaes_in_keras%2Fbinarized_mnist.png)

#### CIFAR-10 Dataset ####

Extract the compressed file `CIFAR_daset.zip`.
A folder named `CIFAR_dataset` should appear, which contains the TRAIN and TEST files of the CIFAR-10 and CIFAR-100 dataset, along with their labels. The CIFAR-10 dataset contains data from 10 classes, while the CIFAR-100 dataset contains data from 100 classes.

##### VAE in Keras Grayscale output images
![cifar10_grayscale.png](output_images%2Fvaes_in_keras%2Fcifar10_grayscale.png)

##### VAE in Keras RGB output images
![cifar10_rgb.png](output_images%2Fvaes_in_keras%2Fcifar10_rgb.png)

#### OMNIGLOT Dataset ####

Extract the compressed file `OMNIGLOT_daset.zip`.
A folder named `OMNIGLOT_dataset` should appear, which contains the TRAIN and TEST files of the OMNIGLOT dataset, from 50 different alphabets, along with their labels.
Two alphabets are used, the Greek and the English.
The Greek alphabet has 24 characters, which means 24 are the classes.
The Greek alphabet has 26 characters, which means 26 are the classes.
The classes are not important for the algorithm, but they are used for plotting purposes.

##### VAE in Keras English output images
![omniglot_english.png](output_images%2Fvaes_in_keras%2Fomniglot_english.png)

##### VAE in Keras Greek output images
![omniglot_greek.png](output_images%2Fvaes_in_keras%2Fomniglot_greek.png)

# K-NN Missing Values completion algorithm #

These are implementations of K-NN Missing Values algorithms on various datasets with missing values.
The datasets included are: MNIST, Binarized MNIST, CIFAR-10 and OMNIGLOT.
I've implemented an algorithm that uses K-NN for regression, i.e. it predicts the missing pixel values, based on the corresponding pixels of the top K nearest neighbors.

### MNIST Dataset ###

The results of the algorithm will be new images of digits, with their missing halves predicted.

#### Output images
|                                    Original Data                                     |                                                         Data with Structured Missing Values K=10                                                         |                                               Predicted Test Data K=10                                               |
|:------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|
| ![Test Data.png](output_images%2Fknn_missing_values%2Fmnist%2FTest%20Data.png)       |        ![Test Data with Missing Values K=10.png](output_images%2Fknn_missing_values%2Fmnist%2FTest%20Data%20with%20Missing%20Values%20K%3D10.png)        | ![Predicted Test Data K=10.png](output_images%2Fknn_missing_values%2Fmnist%2FPredicted%20Test%20Data%20K%3D10.png)   |

### Binarized MNIST Dataset ###

The results of the algorithm will be new images of binarized digits, with their missing halves predicted.

#### Output images
|                                                           Original Data                                                            |                                                       Data with Structured Missing Values K=10                                                       |                                                   Predicted Test Data K=10                                                   |
|:----------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------:|
| ![Original Binarized Test Data.png](output_images%2Fknn_missing_values%2Fbinarized_mnist%2FOriginal%20Binarized%20Test%20Data.png) | ![Test Data with Missing Values K=10.png](output_images%2Fknn_missing_values%2Fbinarized_mnist%2FTest%20Data%20with%20Missing%20Values%20K%3D10.png) | ![Predicted Test Data K=10.png](output_images%2Fknn_missing_values%2Fbinarized_mnist%2FPredicted%20Test%20Data%20K%3D10.png) |

### CIFAR-10 Dataset ###

The results of the algorithm will be new images of the selected category (e.g. cats, dogs, etc.),
with their missing halves predicted.

### OMNIGLOT Dataset ###

The results of the algorithm will be new images of alphabet characters, with their missing halves predicted.

### ORL Face Database Dataset ###

### How to set up and run the K-NN Missing Values algorithm on the ORL Face Database dataset ###
Extract the compressed file `ORL_Face_Dataset.zip`.
Create a folder named `ORL_Face_Dataset` and unzip there the contents of the zip file.
In the dataset, there are 400 face images in total, from 40 different persons and 10 images from each person, 40 * 10 = 400.
The results of the algorithm will be new images of the faces, with their missing halves predicted.


# VAE Missing Values completion algorithm #

There are also two different implementations of the Variational Autoencoder Missing Values algorithm of the VAEs included.

1. using TensorFlow and 
2. using PyTorch

The datasets included are: MNIST, Binarized MNIST, CIFAR-10, OMNIGLOT, ORL Face Database, Yale Faces & the Movielens dataset.
The algorithm uses a Variational Autoencoder to predict only the missing pixel values, based on the training data.

**Note:** In some datasets, e.g. in the CIFAR-10 dataset, the results are good only if the images are grayscaled!

#### VAE Missing Values completion algorithm in PyTorch MNIST Dataset
|                                          Original data                                           |                              Data with Structured Missing Values                               |                                        Epoch 200                                         |
|:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|
| ![original_data.png](output_images%2Fvaes_missing_values_in_pytorch%2Fmnist%2Foriginal_data.png) | ![missing_data.png](output_images%2Fvaes_missing_values_in_pytorch%2Fmnist%2Fmissing_data.png) | ![epoch_200.png](output_images%2Fvaes_missing_values_in_pytorch%2Fmnist%2Fepoch_200.png) |

#### VAE Missing Values completion algorithm in PyTorch OMNIGLOT English Dataset
|                                                                Original data                                                                |                                                      Data with Random Missing Values                                                      |                                                              Epoch 100                                                              |
|:-------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|
| ![original_data_characters_1-10.png](output_images%2Fvaes_missing_values_in_pytorch%2Fomniglot_english%2Foriginal_data_characters_1-10.png) | ![missing_data_characters_1-10.png](output_images%2Fvaes_missing_values_in_pytorch%2Fomniglot_english%2Fmissing_data_characters_1-10.png) | ![epoch_100_characters_1-10.png](output_images%2Fvaes_missing_values_in_pytorch%2Fomniglot_english%2Fepoch_100_characters_1-10.png) |

#### VAE Missing Values completion algorithm in PyTorch OMNIGLOT Greek Dataset
|                                                               Original data                                                               |                                                     Data with Random Missing Values                                                      |                                                             Epoch 100                                                             |
|:-----------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|
| ![original_data_characters_1-10.png](output_images%2Fvaes_missing_values_in_pytorch%2Fomniglot_greek%2Foriginal_data_characters_1-10.png) | ![missing_data_characters_1-10.png](output_images%2Fvaes_missing_values_in_pytorch%2Fomniglot_greek%2Fmissing_data_characters_1-10.png)  | ![epoch_100_characters_1-10.png](output_images%2Fvaes_missing_values_in_pytorch%2Fomniglot_greek%2Fepoch_100_characters_1-10.png) |

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

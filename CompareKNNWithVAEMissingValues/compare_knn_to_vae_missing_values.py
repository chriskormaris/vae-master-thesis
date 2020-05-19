from Utilities import *
import pandas as pd
import numpy as np


missing_value = 0

mask_df = pd.read_csv('users_movies_ratings_mask.csv', sep='\t', header=None)
mask_df = mask_df.values

non_missing_values_percentage = np.count_nonzero(mask_df) / np.size(mask_df) * 100
print('non missing values percentage: ' + str(non_missing_values_percentage) + ' %')

print('')

knn_predicted_values_df = pd.read_csv('users_movies_ratings_knn_predicted_values.csv', sep='\t', header=None)
knn_predicted_values_df = knn_predicted_values_df.replace(to_replace='---', value=1.0)
knn_predicted_values_df = knn_predicted_values_df.astype(float)

# convert to numpy matrix
knn_predicted_values = knn_predicted_values_df.values

no_users = knn_predicted_values.shape[0]
no_movies = knn_predicted_values.shape[1]

print('number of users: ' + str(no_users))
print('number of movies: ' + str(no_movies))

knn_predicted_values_df.index = range(1, no_users+1)
knn_predicted_values_df.columns = range(1, no_movies+1)
# print(knn_predicted_values_df.head(2))

# print('')

vae_pytorch_predicted_values_df = pd.read_csv('users_movies_ratings_vae_pytorch_predicted_values.csv', sep='\t', header=None)
vae_pytorch_predicted_values_df = vae_pytorch_predicted_values_df.replace(to_replace='---', value=1.0)
vae_pytorch_predicted_values_df = vae_pytorch_predicted_values_df.astype(float)
# round dataframe
vae_pytorch_predicted_values_df = vae_pytorch_predicted_values_df.round()
vae_pytorch_predicted_values_df.index = range(1, no_users + 1)
vae_pytorch_predicted_values_df.columns = range(1, no_movies + 1)
# print(vae_pytorch_predicted_values_df.head(2))

# convert to numpy matrix
vae_pytorch_predicted_values = vae_pytorch_predicted_values_df.values

print('')

print('Comparison between K-NN and VAE in PyTorch Missing Values algorithm')

error1 = rmse(knn_predicted_values, vae_pytorch_predicted_values)
print('root mean squared error: ' + str(error1))

error2 = mae(knn_predicted_values, vae_pytorch_predicted_values)
print('mean absolute error: ' + str(error2))

match = np.sum(np.equal(vae_pytorch_predicted_values, knn_predicted_values)) / np.size(vae_pytorch_predicted_values) * 100
print('match percentage = ' + str(match) + ' %')

indices = np.where(knn_predicted_values != missing_value)
print('knn algorithm mean rating: ' + str(knn_predicted_values[indices].mean()))
indices = np.where(vae_pytorch_predicted_values != missing_value)
print('vae in pytorch mean rating: ' + str(vae_pytorch_predicted_values[indices].mean()))

print('')

vae_tensorflow_predicted_values_df = pd.read_csv('users_movies_ratings_vae_tensorflow_predicted_values.csv', sep='\t', header=None)
vae_tensorflow_predicted_values_df = vae_tensorflow_predicted_values_df.replace(to_replace='---', value=1.0)
vae_tensorflow_predicted_values_df = vae_tensorflow_predicted_values_df.astype(float)
# round dataframe
vae_tensorflow_predicted_values_df = vae_tensorflow_predicted_values_df.round()
vae_tensorflow_predicted_values_df.index = range(1, no_users + 1)
vae_tensorflow_predicted_values_df.columns = range(1, no_movies + 1)
# print(vae_tensorflow_predicted_values_df.head(2))

# convert to numpy matrix
vae_tensorflow_predicted_values = vae_tensorflow_predicted_values_df.values

print('')

print('Comparison between VAE in PyTorch and VAE in TensorFlow Missing Values algorithm')

error1 = rmse(vae_pytorch_predicted_values, vae_tensorflow_predicted_values)
print('root mean squared error: ' + str(error1))

error2 = mae(vae_pytorch_predicted_values, vae_tensorflow_predicted_values)
print('mean absolute error: ' + str(error2))

match = np.sum(np.equal(vae_pytorch_predicted_values, vae_tensorflow_predicted_values)) / np.size(vae_pytorch_predicted_values) * 100
print('match = ' + str(match) + ' %')

indices = np.where(vae_pytorch_predicted_values != missing_value)
print('vae in pytorch mean rating: ' + str(vae_pytorch_predicted_values[indices].mean()))
indices = np.where(vae_tensorflow_predicted_values != missing_value)
print('vae in tensorflow mean rating: ' + str(vae_tensorflow_predicted_values[indices].mean()))

print('')
print('')

print('Comparison between K-NN and VAE in TensorFlow Missing Values algorithm')

error1 = rmse(knn_predicted_values, vae_tensorflow_predicted_values)
print('root mean squared error: ' + str(error1))

error2 = mae(knn_predicted_values, vae_tensorflow_predicted_values)
print('mean absolute error: ' + str(error2))

match = np.sum(np.equal(knn_predicted_values, vae_tensorflow_predicted_values)) / np.size(vae_pytorch_predicted_values) * 100
print('match = ' + str(match) + ' %')

indices = np.where(knn_predicted_values != missing_value)
print('knn algorithm mean rating: ' + str(knn_predicted_values[indices].mean()))
indices = np.where(vae_tensorflow_predicted_values != missing_value)
print('vae in tensorflow mean rating: ' + str(vae_tensorflow_predicted_values[indices].mean()))

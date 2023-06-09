import pandas as pd

from utilities import *

missing_value = 0

mask_df = pd.read_csv('users_movies_ratings_mask.csv', sep='\t', header=None)
mask_df = mask_df.values

non_missing_values_percentage = np.count_nonzero(mask_df) / np.size(mask_df) * 100
print(f'non missing values percentage: {non_missing_values_percentage} %')

print()

knn_predicted_values_df = pd.read_csv('users_movies_ratings_knn_predicted_values.csv', sep='\t', header=None)
knn_predicted_values_df = knn_predicted_values_df.replace(to_replace='---', value=1.0)
knn_predicted_values_df = knn_predicted_values_df.astype(float)

# convert to numpy matrix
knn_predicted_values = knn_predicted_values_df.values

num_users = knn_predicted_values.shape[0]
num_movies = knn_predicted_values.shape[1]

print(f'number of users: {num_users}')
print(f'number of movies: {num_movies}')

knn_predicted_values_df.index = range(1, num_users + 1)
knn_predicted_values_df.columns = range(1, num_movies + 1)
# print(knn_predicted_values_df.head(2))

# print()

vae_pytorch_predicted_values_df = pd.read_csv(
    'users_movies_ratings_vae_pytorch_predicted_values.csv',
    sep='\t',
    header=None
)
vae_pytorch_predicted_values_df = vae_pytorch_predicted_values_df.replace(to_replace='---', value=1.0)
vae_pytorch_predicted_values_df = vae_pytorch_predicted_values_df.astype(float)
# round dataframe
vae_pytorch_predicted_values_df = vae_pytorch_predicted_values_df.round()
vae_pytorch_predicted_values_df.index = range(1, num_users + 1)
vae_pytorch_predicted_values_df.columns = range(1, num_movies + 1)
# print(vae_pytorch_predicted_values_df.head(2))

# convert to numpy matrix
vae_pytorch_predicted_values = vae_pytorch_predicted_values_df.values

print()

print('Comparison between K-NN and VAE in PyTorch Missing Values algorithm')

error1 = rmse(knn_predicted_values, vae_pytorch_predicted_values)
print(f'root mean squared error: {error1}')

error2 = mae(knn_predicted_values, vae_pytorch_predicted_values)
print(f'mean absolute error: {error2}')

match = np.sum(np.equal(vae_pytorch_predicted_values, knn_predicted_values)) / \
        np.size(vae_pytorch_predicted_values) * 100
print(f'match percentage = {match} %')

indices = np.where(knn_predicted_values != missing_value)
print(f'knn algorithm mean rating: {knn_predicted_values[indices].mean()}')
indices = np.where(vae_pytorch_predicted_values != missing_value)
print(f'vae in pytorch mean rating: {vae_pytorch_predicted_values[indices].mean()}')

print()

vae_tensorflow_predicted_values_df = pd.read_csv(
    '../movielens_output_data/VAEsMissingValuesInTensorFlow/users_movies_ratings_predicted_values.csv',
    sep='\t',
    header=None
)
vae_tensorflow_predicted_values_df = vae_tensorflow_predicted_values_df.replace(to_replace='---', value=1.0)
vae_tensorflow_predicted_values_df = vae_tensorflow_predicted_values_df.astype(float)
# round dataframe
vae_tensorflow_predicted_values_df = vae_tensorflow_predicted_values_df.round()
vae_tensorflow_predicted_values_df.index = range(1, num_users + 1)
vae_tensorflow_predicted_values_df.columns = range(1, num_movies + 1)
# print(vae_tensorflow_predicted_values_df.head(2))

# convert to numpy matrix
vae_tensorflow_predicted_values = vae_tensorflow_predicted_values_df.values

print()

print('Comparison between VAE in PyTorch and VAE in TensorFlow Missing Values algorithm')

error1 = rmse(vae_pytorch_predicted_values, vae_tensorflow_predicted_values)
print(f'root mean squared error: {error1}')

error2 = mae(vae_pytorch_predicted_values, vae_tensorflow_predicted_values)
print(f'mean absolute error: {error2}')

match = np.sum(np.equal(vae_pytorch_predicted_values, vae_tensorflow_predicted_values)) / \
        np.size(vae_pytorch_predicted_values) * 100
print(f'match = {match} %')

indices = np.where(vae_pytorch_predicted_values != missing_value)
print(f'vae in pytorch mean rating: {vae_pytorch_predicted_values[indices].mean()}')
indices = np.where(vae_tensorflow_predicted_values != missing_value)
print(f'vae in tensorflow mean rating: {vae_tensorflow_predicted_values[indices].mean()}')

print('\n')

print('Comparison between K-NN and VAE in TensorFlow Missing Values algorithm')

error1 = rmse(knn_predicted_values, vae_tensorflow_predicted_values)
print(f'root mean squared error: {error1}')

error2 = mae(knn_predicted_values, vae_tensorflow_predicted_values)
print(f'mean absolute error: {error2}')

match = np.sum(np.equal(knn_predicted_values, vae_tensorflow_predicted_values)) / \
        np.size(vae_pytorch_predicted_values) * 100
print(f'match = {match} %')

indices = np.where(knn_predicted_values != missing_value)
print(f'knn algorithm mean rating: {knn_predicted_values[indices].mean()}')
indices = np.where(vae_tensorflow_predicted_values != missing_value)
print(f'vae in tensorflow mean rating: {vae_tensorflow_predicted_values[indices].mean()}')

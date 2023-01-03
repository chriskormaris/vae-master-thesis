from importlib.machinery import SourceFileLoader


get_binarized_mnist_dataset = SourceFileLoader(
    'get_binarized_mnist_dataset',
    'Utilities/get_binarized_mnist_dataset.py'
).load_module()

get_cifar10_dataset = SourceFileLoader('get_cifar10_dataset', 'Utilities/get_cifar10_dataset.py').load_module()

get_omniglot_dataset = SourceFileLoader('get_omniglot_dataset', 'Utilities/get_omniglot_dataset.py').load_module()

get_orl_faces_dataset = SourceFileLoader('get_orl_faces_dataset', 'Utilities/get_orl_faces_dataset.py').load_module()

get_yale_faces_dataset = SourceFileLoader('get_yale_faces_dataset', 'Utilities/get_yale_faces_dataset.py').load_module()

get_movielens_dataset = SourceFileLoader('get_movielens_dataset', 'Utilities/get_movielens_dataset.py').load_module()

kNN = SourceFileLoader('kNN', 'Utilities/kNN_matrix_completion.py').load_module()

plot_dataset_samples = SourceFileLoader('plot_dataset_samples', 'Utilities/plot_dataset_samples.py').load_module()

Utilities = SourceFileLoader('Utilities', 'Utilities/Utilities.py').load_module()

vae_in_keras = SourceFileLoader('vae_in_keras', 'Utilities/vae_in_keras.py').load_module()

vae_in_tensorflow = SourceFileLoader('vae_in_tensorflow', 'Utilities/vae_in_tensorflow.py').load_module()

initialize_weights_in_pytorch = SourceFileLoader(
    'initialize_weights_in_pytorch',
    'Utilities/initialize_weights_in_pytorch.py'
).load_module()

vae_in_pytorch = SourceFileLoader('vae_in_pytorch', 'Utilities/vae_in_pytorch.py').load_module()

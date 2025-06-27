import os

version = '3.0.0'
author = 'Christos Kormaris'

icons_path = 'icons'
output_img_base_path = 'output_images'
movielens_output_data_base_path = 'movielens_output_data'
tensorflow_logs_path = 'tensorflow_logs'
save_base_path = 'save'

datasets_base_path = os.path.join('..', 'DATASETS')

binarized_dataset_path = os.path.join(datasets_base_path, 'Binarized_MNIST_dataset')
movielens_dataset_path = os.path.join(datasets_base_path, 'MovieLens_dataset', 'ml-100k')
omniglot_dataset_path = os.path.join(datasets_base_path, 'OMNIGLOT_dataset')
orl_faces_dataset_path = os.path.join(datasets_base_path, 'ORL_Face_dataset')
yale_faces_dataset_path = os.path.join(datasets_base_path, 'YALE_dataset', 'CroppedYale')

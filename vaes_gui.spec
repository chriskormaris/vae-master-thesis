# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['vaes_gui.py'],
             pathex=['.'],
             binaries=[],
             datas=[('__init__.py', './'),
                    ('icons\\aueb_logo.png', 'icons'),
                    ('icons\\help.ico', 'icons'),
                    ('icons\\info.ico', 'icons'),
                    ('icons\\vaes.ico', 'icons'),
                    ('Utilities\\get_binarized_mnist_dataset.py', 'Utilities'),
                    ('Utilities\\get_cifar10_dataset.py', 'Utilities'),
                    ('Utilities\\get_movielens_dataset.py', 'Utilities'),
                    ('Utilities\\get_omniglot_dataset.py', 'Utilities'),
                    ('Utilities\\get_orl_faces_dataset.py', 'Utilities'),
                    ('Utilities\\get_yale_faces_dataset.py', 'Utilities'),
                    ('Utilities\\initialize_weights_in_pytorch.py', 'Utilities'),
                    ('Utilities\\kNN_matrix_completion.py', 'Utilities'),
                    ('Utilities\\plot_dataset_samples.py', 'Utilities'),
                    ('Utilities\\Utilities.py', 'Utilities'),
                    ('Utilities\\vae_in_keras.py', 'Utilities'),
                    ('Utilities\\vae_in_pytorch.py', 'Utilities'),
                    ('Utilities\\vae_in_tensorflow.py', 'Utilities'),
                    ('VAEsInKeras\\binarized_mnist.py', 'VAEsInKeras'),
                    ('VAEsInKeras\\cifar10.py', 'VAEsInKeras'),
                    ('VAEsInKeras\\mnist.py', 'VAEsInKeras'),
                    ('VAEsInKeras\\omniglot.py', 'VAEsInKeras'),
                    ('VAEsInKeras\\orl_faces.py', 'VAEsInKeras'),
                    ('VAEsInKeras\\yale_faces.py', 'VAEsInKeras'),
                    ('VAEsInPyTorch\\binarized_mnist.py', 'VAEsInPyTorch'),
                    ('VAEsInPyTorch\\cifar10.py', 'VAEsInPyTorch'),
                    ('VAEsInPyTorch\\mnist.py', 'VAEsInPyTorch'),
                    ('VAEsInPyTorch\\omniglot.py', 'VAEsInPyTorch'),
                    ('VAEsInPyTorch\\orl_faces.py', 'VAEsInPyTorch'),
                    ('VAEsInPyTorch\\yale_faces.py', 'VAEsInPyTorch'),
                    ('VAEsInTensorFlow\\binarized_mnist.py', 'VAEsInTensorFlow'),
                    ('VAEsInTensorFlow\\cifar10.py', 'VAEsInTensorFlow'),
                    ('VAEsInTensorFlow\\mnist.py', 'VAEsInTensorFlow'),
                    ('VAEsInTensorFlow\\omniglot.py', 'VAEsInTensorFlow'),
                    ('VAEsInTensorFlow\\orl_faces.py', 'VAEsInTensorFlow'),
                    ('VAEsInTensorFlow\\yale_faces.py', 'VAEsInTensorFlow'),
                    ('kNNMissingValues\\binarized_mnist.py', 'kNNMissingValues'),
                    ('kNNMissingValues\\cifar10.py', 'kNNMissingValues'),
                    ('kNNMissingValues\\mnist.py', 'kNNMissingValues'),
                    ('kNNMissingValues\\movielens.py', 'kNNMissingValues'),
                    ('kNNMissingValues\\omniglot.py', 'kNNMissingValues'),
                    ('kNNMissingValues\\orl_faces.py', 'kNNMissingValues'),
                    ('kNNMissingValues\\yale_faces.py', 'kNNMissingValues'),
                    ('VAEsMissingValuesInPyTorch\\binarized_mnist.py', 'VAEsMissingValuesInPyTorch'),
                    ('VAEsMissingValuesInPyTorch\\cifar10.py', 'VAEsMissingValuesInPyTorch'),
                    ('VAEsMissingValuesInPyTorch\\mnist.py', 'VAEsMissingValuesInPyTorch'),
                    ('VAEsMissingValuesInPyTorch\\movielens.py', 'VAEsMissingValuesInPyTorch'),
                    ('VAEsMissingValuesInPyTorch\\omniglot.py', 'VAEsMissingValuesInPyTorch'),
                    ('VAEsMissingValuesInPyTorch\\orl_faces.py', 'VAEsMissingValuesInPyTorch'),
                    ('VAEsMissingValuesInPyTorch\\yale_faces.py', 'VAEsMissingValuesInPyTorch'),
                    ('VAEsMissingValuesInTensorFlow\\binarized_mnist.py', 'VAEsMissingValuesInTensorFlow'),
                    ('VAEsMissingValuesInTensorFlow\\cifar10.py', 'VAEsMissingValuesInTensorFlow'),
                    ('VAEsMissingValuesInTensorFlow\\mnist.py', 'VAEsMissingValuesInTensorFlow'),
                    ('VAEsMissingValuesInTensorFlow\\movielens.py', 'VAEsMissingValuesInTensorFlow'),
                    ('VAEsMissingValuesInTensorFlow\\omniglot.py', 'VAEsMissingValuesInTensorFlow'),
                    ('VAEsMissingValuesInTensorFlow\\orl_faces.py', 'VAEsMissingValuesInTensorFlow'),
                    ('VAEsMissingValuesInTensorFlow\\yale_faces.py', 'VAEsMissingValuesInTensorFlow')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['Tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='vaes_gui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          icon='icons\\vaes.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='vaes_gui')

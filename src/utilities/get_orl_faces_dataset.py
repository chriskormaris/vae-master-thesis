import os

import numpy as np
from PIL import Image


# The dataset contains images 400 images of faces.
# The faces are from 40 different people.
# 10 faces for each person are included, thus 40 * 10 = 400.
def get_orl_faces_dataset(db_of_faces_path, one_hot=False, print_progress=False):
    D = 10304  # 92 x 112
    K = 40  # number of classes

    X = np.array([[]])
    y = np.array([])

    subdirectories = sorted(os.listdir(db_of_faces_path))

    i = 0
    for subdir in subdirectories:
        subdir_path = db_of_faces_path + '/' + subdir
        if os.path.isdir(subdir_path):
            if print_progress:
                print('Reading images from directory "' + subdir + '"...')
            files = sorted(os.listdir(subdir_path))
            for filename in files:
                k = 0  # counter for the pictures in the subdir
                try:
                    if filename.split('.')[1] == 'pgm':
                        im = Image.open(subdir_path + '/' + filename)
                        image = np.array(im)
                        if image.size == D:
                            image = image.reshape((1, D))
                            if X.size == 0:
                                X = image
                            else:
                                X = np.concatenate((X, image), axis=0)
                        k = k + 1
                except IndexError:
                    pass
                if y.size == 0:
                    y = np.zeros((k,), dtype=np.int8)
                else:
                    y = np.concatenate((y, i * np.ones((k,), dtype=np.int8)), axis=0)
            i = i + 1

    # We will normalize all values between 0 and 1,.
    X = X.astype('float32') / 255.

    if one_hot:
        t = np.zeros((y.size, K))
        t[np.arange(y.size), y] = 1
        return X, t
    else:
        return X, y

import inspect
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from __init__ import *


__author__ = 'c.kormaris'

missing_value = 0.5
K = int(sys.argv[1])


###############

# MAIN #

if __name__ == '__main__':

        yale_faces_dataset_dir = '../YALE_dataset/CroppedYale'

        output_images_dir = './output_images/kNNMissingValues/yale_faces'

        if not os.path.exists(output_images_dir):
                os.makedirs(output_images_dir)

        num_classes = 10

        # LOAD YALE FACES DATASET #
        print('Getting YALE faces dataset...')
        X, y = get_yale_faces_dataset.get_yale_faces_dataset(yale_faces_dataset_dir)

        # construct data with missing values
        X_missing, X, y = Utilities.construct_missing_data(X, y, structured_or_random=sys.argv[2])

        # plot original data X
        fig = plot_dataset_samples.plot_yale_faces(X, y, categories=list(range(10)), show_plot=False)
        fig.savefig(output_images_dir + '/Original Faces ' + str(1) + '-' + str(10) + ' K=' + str(K) + '.png', bbox_inches='tight')
        plt.close()

        # plot data with missing values
        fig = plot_dataset_samples.plot_yale_faces(X_missing, y, categories=list(range(10)), show_plot=False)
        fig.savefig(output_images_dir + '/Faces ' + str(1) + '-' + str(10) + ' with Mixed Missing Values K=' + str(K) + '.png', bbox_inches='tight')
        plt.close()

        # Compute how sparse is the matrix X_train. Print the percentage of non-missing entries compared to the total entries of the matrix.
        percentage = Utilities.non_missing_percentage(X_missing)
        print('non missing values percentage: ' + str(percentage) + ' %')

        # convert variables to numpy matrices
        X = np.array(X)
        X_missing = np.array(X_missing)

        print('')

        # run K-NN
        print('Running %i-NN algorithm...' % K)
        print('')

        start_time = time.time()
        X_predicted = kNN.kNNMatrixCompletion(X, X_missing, K, missing_value)
        elapsed_time = time.time() - start_time

        print('k-nn predictions calculations time: ' + str(elapsed_time))
        print('')

        # plot predicted data
        fig = plot_dataset_samples.plot_yale_faces(X_predicted, y, categories=list(range(10)), show_plot=False)
        fig.savefig(output_images_dir + '/Predicted Faces ' + str(1) + '-' + str(10) + ' K=' + str(K) + '.png', bbox_inches='tight')
        plt.close()

        error1 = Utilities.rmse(X, X_predicted)
        print('root mean squared error: ' + str(error1))

        error2 = Utilities.mae(X, X_predicted)
        print('mean absolute error: ' + str(error2))

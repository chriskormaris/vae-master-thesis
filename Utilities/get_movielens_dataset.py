import numpy as np
import pandas as pd


def get_movielens_data(movielens_dataset_dir):
    # TRAIN data
    dataframe = pd.read_csv(movielens_dataset_dir + '/ua.base',
                            sep='\t', header=None, names=['userID', 'movieId', 'rating', 'col3'])
    dataframe['userID'] = dataframe['userID'] - 1
    dataframe['movieId'] = dataframe['movieId'] - 1
    del dataframe['col3']

    dataframe = dataframe.sort_values(['userID', 'movieId'])
    dataframe.index = list(range(len(dataframe)))

    X_train = np.zeros((np.max(dataframe['userID']) + 1, np.max(dataframe['movieId']) + 1))
    X_train[dataframe['userID'], dataframe['movieId']] = dataframe['rating'].apply(pd.to_numeric)

    # TEST data
    dataframe_test = pd.read_csv(movielens_dataset_dir + '/ua.test',
                            sep='\t', header=None, names=['userID', 'movieId', 'rating', 'col3'])
    dataframe_test['userID'] = dataframe_test['userID'] - 1
    dataframe_test['movieId'] = dataframe_test['movieId'] - 1
    del dataframe_test['col3']

    dataframe_test = dataframe_test.sort_values(['userID', 'movieId'])
    dataframe_test.index = list(range(len(dataframe_test)))

    X_test = np.zeros((np.max(dataframe_test['userID']) + 1, np.max(dataframe_test['movieId']) + 1))
    X_test[dataframe_test['userID'], dataframe_test['movieId']] = dataframe_test['rating'].apply(pd.to_numeric)

    # ALL data
    dataframe_merged = pd.DataFrame(pd.concat((dataframe, dataframe_test), join='inner'))
    dataframe_merged = dataframe_merged.sort_values(['userID', 'movieId'])
    dataframe_merged.index = list(range(len(dataframe_merged)))

    X_all = np.zeros((np.max(dataframe_merged['userID']) + 1, np.max(dataframe_merged['movieId']) + 1))
    X_all[dataframe_merged['userID'], dataframe_merged['movieId']] = dataframe_merged['rating'].apply(pd.to_numeric)

    return X_train, X_test, X_all


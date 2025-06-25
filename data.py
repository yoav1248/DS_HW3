import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

np.random.seed(42)


class StandardScaler:
    def __init__(self):
        """ object instantiation """
        self.mean = None
        self.std = None

    def fit(self, X):
        """ fit scaler by learning the mean and standard deviation per feature """
        self.mean = X.mean()
        self.std = X.std(ddof=1)

    def transform(self, X):
        """ transform X by learned mean and standard deviation, and return it """
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        """ fit scaler by learning the mean and standard deviation per feature, and then transform X """
        self.fit(X)
        return self.transform(X)


def load_data(path):
    """ reads and returns the pandas DataFrame """
    return pd.read_csv(path)


def adjust_labels(y):
    """adjust labels of season from {0,1,2,3} to {0,1}"""
    return y // 2


def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.0001^2)
    """
    noise = np.random.normal(loc=0, scale=0.0001, size=data.shape)
    return data + noise


def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=42)

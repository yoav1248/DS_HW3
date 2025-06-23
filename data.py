import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
np.random.seed(42)


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

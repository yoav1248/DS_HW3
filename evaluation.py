import numpy as np
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    """ returns f1_score of binary classification task
     with true labels y_true and predicted labels y_pred """

    n_tp = np.sum(y_true & y_pred)
    recall = n_tp / np.sum(y_true)
    precision = n_tp / np.sum(y_pred)

    return 2 * recall * precision / (recall + precision)


def rmse(y_true, y_pred):
    """returns RMSE of regression task
    with true labels y_true and predicted labels y_pred"""

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def visualize_results(k_list, scores, metric, title, path):
    """ plot a results graph of cross validation scores """
    pass

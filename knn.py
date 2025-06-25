import numpy as np
from statistics import mode
from abc import abstractmethod, ABC
from data import StandardScaler


class KNN(ABC):
    def __init__(self, k):
        """ object instantiation, save k and define a scaler object """
        self.k = k
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """ fit scaler and save X_train and y_train """
        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train.copy()

    @abstractmethod
    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        pass

    def neighbours_indices(self, x):
        """ for a given point x, find indices of k closest points in the training set """
        x = self.scaler.transform(x)
        return KNN.dist(self.X_train, x).argsort()[:self.k]

    @staticmethod
    def dist(x1, x2):
        """returns Euclidean distance between x1 and x2"""
        return np.linalg.norm(x1 - x2, axis=-1)


class ClassificationKnn(KNN):
    def __init__(self, k):
        """ object instantiation, parent class instantiation """
        super().__init__(k)

    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        get_predicted_value = lambda x: mode(self.y_train[self.neighbours_indices(x)])
        return np.array([get_predicted_value(x) for x in X_test])


class RegressionKnn(KNN):
    def __init__(self, k):
        """ object instantiation, parent class instantiation """
        super().__init__(k)

    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        get_predicted_value = lambda x: self.y_train[self.neighbours_indices(x)].mean()
        return np.array([get_predicted_value(x) for x in X_test])

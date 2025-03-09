from __future__ import print_function, division
import numpy as np
import math
from myenv. utils. misc import  Plot
from myenv. utils. data_manipulation import make_diagonal

class  Sigmoid:
    def __call__(self, X):
        return 1 / (1 + np.exp(-X))

    def gradient(self, X):
        return  self.__call__(X) * (1 - self.__call__(X))

class LogisticRegression:

    def __init__(self, learning_rate=.1, gradient_descent=True):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def _initialize_parameter(self, X):
        n_features = np.shape(X)[1]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameter(X)
        for i in range(n_iterations):
            y_pred = self.sigmoid(X.dot(self.param))
            if self.gradient_descent:
                self.param -= self.learning_rate * -(y - y_pred).dot(X)
            else:
                diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.param)))

                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param + y - y_pred))

    def predict(self,X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred





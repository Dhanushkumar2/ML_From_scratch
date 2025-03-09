from __future__ import print_function, division
import numpy as np
from myenv. utils. data_operation import calculate_correlation_matrix
from myenv. utils. data_manipulation import normalize, standardize

class LDA:
    def __init__(self):
        self.w = None

    def transform(self, X, y):
        self.fit(X, y)
        X_transform = X.dot(self.w)
        return X_transform

    def fit(self,X ,y):
        X1 = X[y == 0]
        X2 = X[y == 1]

        cov1 = calculate_correlation_matrix(X1)
        cov2 = calculate_correlation_matrix(X2)
        cov_tot = cov1 + cov2

        mean1 = X1.mean(0)
        mean2 = X2.mean(0)
        mean_diff = np.atleast_1d(mean1 - mean2)

        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)


    def predict(self, X):
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred

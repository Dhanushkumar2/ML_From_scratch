from __future__ import print_function, division

import math
from typing import Any

import numpy as np
from pyexpat import features
import utils
from numpy import ndarray, dtype, floating

from myenv.utils import normalize, polynomial_feature


class l1_regularization():
    # Regularization for lasso Regression

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)

class l2_regularization():
    # Regularization for ridge regression
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)

    def grad(self, w):
        return self.alpha * w

class l1_l2_regularization():
    def __init__(self, alpha, l1_ratio = 0.5):
        self.alpha = alpha
        self.l1_ration = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ration * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ration) * np.linalg.norm(w)
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ration * np.sign(w)
        l2_contr = (1 - self.l1_ration) * w
        return self.alpha * (l1_contr + l2_contr)

class Regression(object):
    """
    Base regression model y = Wx + b 
    """
    
    def __int__(self, n_iterations, learning_rate, regularization=None):
        self.n_iteration = n_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        
    def initialize_weights(self, n_features):
        limit = 1/ math.sqrt(n_features)
        self.w = np.random.uniform (-limit, limit, (n_features, ))
        
    def fit(self, X, y):    #gradient descent
        X = np.insert(X, 0, 1, axis=1)
        self.training_error = []
        self.initialize_weights(n_features=X.shape[1])
        
        for i in range(self.n_iteration):
            y_pred = X.dot(self.w)
            reg_term = self.regularization(self.w) if self.regularization else 0
            mse = np.mean(0.5 * (y - y_pred)**2 ) + reg_term
            self.training_error.append(mse)

            grad_reg = self.regularization(self.w) if self.regularization else 0
            grad_w = -(y - y_pred).dot(X) + grad_reg
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_predict = X.dot(self.w)
        return y_predict

class LinearRegression(Regression):

    def __init__(self, n_iteration=100, learning_rate=0.01, gradient_descent=True):
        self.gradient_descent = gradient_descent
        self.regularization = lambda x:0
        self.regularization.grad = lambda x:0
        super(LinearRegression, self).__init__(n_iteration=n_iteration, learning_rate=learning_rate)

    def fit(self, X, y):
        if not self.gradient_descent:
            X = np.insert(X, 0, 1, axis=1)
            U, S, Vt = np.linalg.svd(X.T.dot(X))
            S_inv = np.diag(S)
            X_sq_reg_inv = Vt.T.dot(S_inv).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)

    def Lasso_Regression(Regression):
        def __init__(self, degree, reg_factor, n_iteration=3000, learning_rate=0.01):
            self.degree = degree
            self.regularization = l1_regularization(alpha=reg_factor)
            super(Lasso_Regression, self).__init__(n_iteration, learning_rate)

        def fit(self, X, y):
            X = normalize(polynomial_feature(X, degree=self.degree))
            super(Lasso_Regression, self).fit(X, y)

        def predict(self, X):
            X = normalize(polynomial_feature(X, degree=self.degree))
            return super(Lasso_Regression, self).predict(X)

class PolynomialRegression(Regression):
    def __init__(self, degree, n_iteration=3000, learning_rate=0.001):
        self.degree = degree
        self.regularization = lambda x:0
        self.regularization.grad = lambda x:0
        super(PolynomialRegression, self).__int__(n_iteration=n_iteration, learning_rate=learning_rate)

    def fit(self, X, y):
        X = polynomial_feature(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X):
        X = polynomial_feature(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X)

    





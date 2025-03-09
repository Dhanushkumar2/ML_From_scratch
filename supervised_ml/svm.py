from __future__ import division, print_function
import pandas as pd
import numpy  as np
from numpy.ma.core import power
import cvxopt
from myenv.utils import train_test_split, normalize, accuracy_score
from myenv.utils.kernels import *
from myenv.utils import Plot

class SVM(object):

    def __init__(self, c=1, kernel=rbf_kernel, power=4, gamma=None, coef=4):
        self.c = c
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multiplier = None
        self.support_vector = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_feature = X.shape

        if not self.gamma:
            self.gamma = 1/ n_feature
        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef= self.coef)

        kernel_matrix = np.zeros((n_samples, n_feature))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc = 'd')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.c:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))

        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.c)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        minmization = cvxopt.solvers.qp(P, q, G, h, A, b)

        lar_mult = np.ravel(minmization['x'])

        idx = lar_mult > 1e-7
        self.lagr_multiplier = lar_mult[idx]
        self.support_vector = X[idx]
        self.support_vector_label = y[idx]

        self.intercept = self.support_vector_label[0]
        for i in range(len(self.lagr_multiplier)):
            self.intercept -= self.lagr_multiplier[i] * self.support_vector_label[i] * self.kernel(self.support_vector[i], self.support_vector[0])


    def predict(self, X):
        y_pred = []
        for sample in X:
            prediction = 0
            for i in range(len(self.lagr_multiplier)):
                prediction += self.lagr_multiplier[i] * self.support_vector_label[i] * self.kernel(self.support_vector[i], sample)

            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)



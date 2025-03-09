from __future__ import division
import numpy as np
import math
import sys


def calculate_entropy(y):
    log2 = lambda x:math.log(x, 2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy

def mean_squared_error(y_true, y_pred):
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse

def calculate_variance(X):
    mean = np.ones(np.shape(X)) * np.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag(X - mean).T.dot(X- mean)

    return variance

def calculate_std_dev(X):
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev

def euclidean_score(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def calculate_covariance_matrix(X, y = None):
    if y is None:
        y = X

    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples -1) * (X - X.mean(axis = 0)).T.dot(y- y.mean(axis=0)))
    return np.array(covariance_matrix, dtype=float)

def calculate_correlation_matrix(X, y=None):
    if y is  None:
        y = X
    n_samples = np.shape(X)[0]
    covariance = (1 / n_samples) * (X - X.mean(0).T.dot(y - y.mean(0)))
    std_dev_x = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(y), 1)
    correlation_matrix = np.divide(covariance, std_dev_x.dot(std_dev_y))

    return np.array(correlation_matrix, dtype=float)

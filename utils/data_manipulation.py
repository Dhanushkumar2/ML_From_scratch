from __future__ import division
from itertools import combinations_with_replacement, combinations
import numpy as np
import math
import sys

from numpy.f2py.auxfuncs import isintent_aux


def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def batch_iterator(X, y=None, batch_size=64):
    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]

def divide_on_feature(X, feature_i, threshold):
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample:sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    x1 = np.array([sample for sample in X if split_func(sample)])
    x2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([x1, x2])

def polynomial_feature(X, degree):
    n_sample, n_feature = np.shape(X)

    def index_combination():
        combs = [combinations_with_replacement(range(n_feature), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs

    combination = index_combination()
    n_output_feature = len(combination)
    x_new = np.empty((n_sample, n_output_feature))

    for i, index_combs in enumerate(combination):
        x_new[:, i] = np.prod(X[:, index_combs], axis = 1)

    return x_new

def get_random_subsets(X, y, n_subsets, replacements=True):
    n_sample = np.shape(X)[0]
    X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(X_y)
    subsets = []

    #Using 50% of training sample without replacement
    subsample_size = int(n_sample // 2)
    if replacements:
        subsample_size = n_sample

    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_sample),
            size = np.shape(range(subsample_size)),
            replace = replacements)
        X = X_y[idx][:, :-1]
        y = X_y[idx][:, :-1]
        subsets.append([X, y])
    return subsets

def normalize(X, axis= -1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def standardize(X):
    X_std = X
    mean = X.mean(axis = 0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col] / std[col])
    return X_std

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    split_i = len(y) - int(len(y)) // (1 / test_size)
    X_train, X_test = X[:, split_i], X[split_i:]
    y_train, y_test = y[:, split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

def k_fold_cross_validation(X, y, k, shuffle=True):
    if shuffle:
        X, y = shuffle_data(X, y)

    n_samples = len(y)
    left_overs = {}
    n_left_overs = (n_samples % k)
    if n_left_overs != 0:
        left_overs["X"] = X[-n_left_overs:]
        left_overs["y"] = y[-n_left_overs:]
        X = X[:-n_left_overs]
        y = y[:-n_left_overs]

    X_split = np.split(X, k)
    y_split = np.split(y, k)
    sets = []
    for i in range(k):
        X_test, y_test = X_split[i], y_split[i]
        X_train = np.concatenate(X_split[:i] + X_split[i + 1:], axis = 0)
        y_train = np.concatenate(y_split[:i] + y_split[i + 1:], axis = 0)
        sets.append([X_train, X_test, y_train, y_test])

    if n_left_overs != 0:
        np.append(sets[-1][0], left_overs["X"], axis = 0)
        np.append(sets[-1][2], left_overs["y"], axis = 0)

    np.array(sets)


def to_categorical(x, n_col=None):
    if n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0], x)] =1
    return one_hot

def to_nominal(x):
    return np.argmax(x, axis = 1)

def make_diagonal(x):
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] =x[i]
        return m

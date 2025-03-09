from __future__ import print_function, division
import numpy as np
from myenv. utils. data_operation import euclidean_score

class KNN:
    def __int__(self, k=5):
        self.k = k

    def _vote(self, neighbor_labels):
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()

    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])
        for i, test_sample in enumerate(X_test):
            idx = np.argsort([euclidean_score(test_sample, x) for x in X_train])[:self.k]
            k_nearest_neighbors = np.array([y_train[j] for j in idx])
            y_pred[i] = self._vote(k_nearest_neighbors)

        return y_pred

import numpy as np
from collections import Counter


# returns the Euclidian distance between p and q
def euclidian_distance(p, q):
    return np.sqrt(np.sum((q - p) ** 2))


class KNN:
    # constructor
    def __init__(self, k):
        self.k = k

    # fits X and y to X_train and y_train
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # predicts multiple samples of data, X
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    # predicts single sample of data, x
    def _predict(self, x):
        # compute distances between unknown data point and all other trained data points
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        # get k nearest sample's labels
        k_indicies = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indicies]
        # get most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

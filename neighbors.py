import numpy as np
from distances import euclidean_distance
from metrics import accuracy, r2

class NNeighborClassifier:

    def __init__(self, dist = euclidean_distance):
        self.dist = dist

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        
        num_test = X.shape[0]
        num_train = self.X.shape[0]
        y = np.zeros(num_test)

        for i in range(num_test):
            min_dist = np.inf
            idx = -1
            for j in range(num_train):
                new_dist = self.dist(X[i], self.X[j])
                if(min_dist > new_dist):
                    min_dist = new_dist
                    idx = j
            y[i] = self.y[idx]
                
        return y

    def score(self, X, y, metric = accuracy):
        y_pred = self.predict(X)
        return metric(y, y_pred)

class NNeighborRegressor:

    def __init__(self, dist = euclidean_distance):
        self.dist = dist

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        
        num_test = X.shape[0]
        num_train = self.X.shape[0]
        y = np.zeros(num_test)

        for i in range(num_test):
            min_dist = np.inf
            idx = -1
            for j in range(num_train):
                new_dist = self.dist(X[i], self.X[j])
                if(min_dist > new_dist):
                    min_dist = new_dist
                    idx = j
            y[i] = self.y[idx]
                
        return y

    def score(self, X, y, metric = r2):
        y_pred = self.predict(X)
        return metric(y, y_pred)

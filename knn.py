

import numpy as np
from numpy.lib.function_base import disp 
from utils import accuracy
import sys
class KNN:
    def __init__(self, n_classes):
        self.n_classes = n_classes
    
    def distance(self, a, b=None, metric=None):
        if not metric:
            metric = 'l2'
        if metric == 'l1':
            return np.sum(np.abs(a - b))
        if metric == 'l2':
            return np.sum((a - b) ** 2)
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X: np.array):
        res = np.zeros((len(X), ))
        for i, xi in enumerate(X):
            dis_xi = np.apply_along_axis(self.distance, arr=self.X, axis=1, b=xi)
            top_k_data = np.argsort(dis_xi)[:self.n_classes]
            # print(self.X.shape, self.y.shape) 
            # print(self.y[top_k_data])
            # sys.exit()
            res[i] = np.argmax(np.bincount(self.y[top_k_data]))
        return res
    
    def score(self, X_test, y_test, metric='accuracy'):
        if not metric:
            metric = 'accuracy'
        y_pred = self.predict(X_test)
        if metric == 'accuracy':
            return accuracy(y_pred, y_test)
            
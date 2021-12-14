
from numpy import random
from numpy.core import numeric
from numpy.lib.function_base import sort_complex
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np 
np.random.seed(1234)


from knn import KNN

def load_data(n_samples=100, n_features=8, n_classes=3):
    X,y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=4,
                            n_redundant=2, n_repeated=0, n_classes=n_classes,
                            n_clusters_per_class=2, weights=None,
                            flip_y=0.01, class_sep=1.0, hypercube=True,
                            shift=0.0, scale=1.0, shuffle=True, random_state=None)
    return X, y

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1234
)

clf = KNN(n_classes=3)
clf.fit(X_train, y_train)
print(clf.predict(X_test))
print(clf.score(X_test, y_test))
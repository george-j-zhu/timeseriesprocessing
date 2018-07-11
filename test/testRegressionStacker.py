from timeseriesprocessing.ensemble.regressionLayer import EnsembleLayer
from timeseriesprocessing.ensemble.regressionStacker import EnsembleStack, EnsembleStackRegressor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

import sklearn

from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from mlxtend.data import boston_housing_data
from mlxtend.plotting import plot_learning_curves

# test code
if __name__ == "__main__":
    # Initializing Classifiers
    clf1 = Lasso(random_state=0)
    clf2 = RandomForestRegressor(random_state=0)
    clf3 = SVR()

    # Creating Stacking
    layer_1 = EnsembleLayer([clf1, clf2, clf3])
    layer_2 = EnsembleLayer([sklearn.clone(clf1)])

    stack = EnsembleStack(cv=3)

    stack.add_layer(layer_1)
    stack.add_layer(layer_2)

    sclf = EnsembleStackRegressor(stack)

    clf_list = [clf1, clf2, clf3, sclf]
    lbl_list = ['Lasso Regression', 'Random Forest', 'RBF kernel SVM', 'Stacking']

    # Loading some example data
    X, y = boston_housing_data()
    X = X[:,[0, 2]]

    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    # Plotting Decision Regions
    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(10, 8))

    itt = itertools.product([0, 1, 2], repeat=2)

    for clf, lab, grd in zip(clf_list, lbl_list, itt):
        clf.fit(X_train, y_train)
        ax = plt.subplot(gs[grd[0], grd[1]])
        plot_learning_curves(X_train, y_train, X_test, y_test, clf)
        plt.title(lab)
    plt.show()
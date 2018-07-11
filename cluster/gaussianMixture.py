#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu

"""
Gaussian Mixture Model
"""

import numpy as np
from sklearn import mixture
from matplotlib import pyplot as plt
import matplotlib as mpl
import itertools
from scipy import linalg


CV_TYPES = ['spherical', 'tied', 'diag', 'full']

class GaussianMixture:
    """
    Gaussian Mixture Model Class for time series processing
    """
    def __init__(self, n_components, max_n_components=None, **params):
        """
        constructor

        Args:
            n_components: number of cluster:
            params: other parameters
        """
        self.n_components = n_components
        self.max_n_components = max_n_components
        if max_n_components is None:
            self.max_n_comonents = n_components
        self.n_components_range = range(1, self.max_n_components + 1)
        self.params = params
        self.bic_ = None
        self.model = None
        self.data = None


    def fit(self, X):
        """
        fit a gaussian mixture model.

        Args:
            X: training data
        """
        bic, best_model = self.__get_bic_scores(X)
        self.model = best_model
        self.bic_ = bic
        self.data = X


    def predict(self, X):
        """
        make predictions.

        Args:
            X: training data
        """
        predictions = self.model.predict(X)
        return predictions


    def fit_predict(self, X):
        """
        fit a guassian mixture model and make predictions.
        """
        self.fit(X)
        predictions = self.model.predict(self.data)
        return predictions

    def get_params(self, deep=True):
        """
        get parameters of this model
        """
        return self.model.get_params(deep)

    def __get_bic_scores(self, X):
        """
        compute bic scores for each covariance type..

        Args:
            X: training data
        """
        lowest_bic = np.infty
        bic = []
        for cv_type in CV_TYPES:
            for n_components in self.n_components_range:
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type)
                gmm.fit(X)
                bic.append(gmm.bic(X))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

        bic = np.array(bic)
        model = best_gmm
        return bic, model

    def plot_bic_scores(self):
        """
        plot BIC scores for each covariance type.
        must be called after fit method.
        """

        color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
        bars = []

        # Plot the BIC scores
        spl = plt.subplot(2, 1, 1)
        for i, (cv_type, color) in enumerate(zip(CV_TYPES, color_iter)):
            xpos = np.array(self.n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, self.bic_[i * len(self.n_components_range):
                                                (i + 1) * len(self.n_components_range)],
                                width=.2, color=color))
        plt.xticks(self.n_components_range)
        plt.ylim([self.bic_.min() * 1.01 - .01 * self.bic_.max(), self.bic_.max()])
        plt.title('BIC score per model')
        xpos = np.mod(self.bic_.argmin(), len(self.n_components_range)) + .65 +\
                    .2 * np.floor(self.bic_.argmin() / len(self.n_components_range))
        plt.text(xpos, self.bic_.min() * 0.97 + .03 * self.bic_.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], CV_TYPES)

        plt.show()

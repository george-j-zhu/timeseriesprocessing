#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu

"""
k-means
"""

import numpy as np
from sklearn.cluster import KMeans as sk_KMeans


class KMeans:
    """
    KMeans class for time series data.
    """

    def __init__(self, n_clusters, max_n_clusters=None, **params):
        """
        constructor

        Args:
            n_clusters: number of cluster:
            params: other parameters
        """
        self.n_clusters = n_clusters
        self.max_n_clusters = max_n_clusters
        self.params = params
        self.model = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.data = None

    def fit(self, X):
        """
        fit the model by the training data.

        Args:
            X: training data
        """
        model = sk_KMeans(n_clusters=self.n_clusters).fit(X)
        self.model = model
        self.cluster_centers_ = model.cluster_centers_
        self.labels_ = model.labels_
        self.inertia_ = model.inertia_
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
        fit and make predictions.

        Args:
            X: training data
        """
        self.fit(X)
        predictions = self.model.predict(self.data)
        return predictions

    def get_params(self, deep=True):
        """
        get parameters of this model
        """
        return self.model.get_params(deep)

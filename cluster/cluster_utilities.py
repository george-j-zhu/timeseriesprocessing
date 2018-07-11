#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu

"""
time series clustering
"""

from sklearn import cluster, mixture
from .gaussianMixture import GaussianMixture
from .kMeans import KMeans
import numpy as np
from matplotlib import pyplot as plt

def get_n_clusters(clusterer):
    """
    get n_clusters
    """
    if isinstance(clusterer, cluster.AffinityPropagation) is True:
        return clusterer.cluster_centers_.shape[0]
    elif isinstance(clusterer, (cluster.AgglomerativeClustering,
                                cluster.Birch, KMeans, cluster.SpectralClustering)):
        return clusterer.n_clusters
    elif isinstance(clusterer, cluster.DBSCAN):
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        return n_clusters_
    elif isinstance(clusterer, (GaussianMixture, mixture.GaussianMixture)):
        return clusterer.n_components
    elif isinstance(clusterer, cluster.MeanShift):
        labels_unique = np.unique(clusterer.labels_)
        n_clusters_ = len(labels_unique)
        return n_clusters_
    else:
        print("clusterer does not contains n_cluster attribute.")
        return None

def plot_elbow_criterion(max_n_clusters, train_x_with_time_df, centroid_based_clusterer):
    """
    plot elbow function for each covariance type.timeSeriesCluster.py

    Args:
        X: training data
        n_cluster: number of cluster
    """

    sse = {}
    for k in range(1, max_n_clusters):
        clusterer = centroid_based_clusterer(n_clusters=k, max_iter=1000).fit(train_x_with_time_df.drop("time", axis=1).values)
        #X["clusters"] = kmeans.labels_
        # print(data["clusters"])
        sse[k] = clusterer.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()
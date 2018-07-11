#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu

import numpy as np
import pandas as pd
from timeseriesprocessing.cluster import timeSeriesCluster
from timeseriesprocessing.cluster import cluster_utilities
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from sklearn.cluster import KMeans as sk_KMeans

def polt_clusters(self, dataframe):
    """
    plot a time series data set with a cluster_id column.
    data set can contain multiple columns, but the first column must be time.
    """
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    for cluster_id in range(self.n_cluster):
        new_dataframe = dataframe[dataframe["cluster_id"] == cluster_id]
        time_array = new_dataframe["time"].values
        actual_data_array = new_dataframe.iloc[:, 1].values
        predicted_data_array = new_dataframe.iloc[:, 2].values
        diff_array = new_dataframe.iloc[:, 3].values
        print(cluster_id)

        fig, ax1 = plt.subplots(figsize=(len(time_array) / 3, 12))
        ax1.scatter(time_array, actual_data_array,
                    label="real data cluster id=" + str(cluster_id),
                    color=colors["violet"])
        ax1.scatter(time_array, predicted_data_array,
                    label="predicted data cluster id=" + str(cluster_id),
                    color="C1")
        ax1.scatter(time_array, diff_array,
                    label="diff cluster id=" + str(cluster_id),
                    color="black")
        ax1.legend(loc=2)
        ax1.set_xticklabels(time_array, rotation=90)
        ax1.tick_params(labelsize=18)
        plt.show()


# test code
if __name__ == "__main__":

    train_x_array = np.array([[1, 2, 5], [1, 4, 5], [1, 0, 5],
                              [4, 2, 5], [4, 4, 5], [4, 0, 5]])

    train_x_df = pd.DataFrame(train_x_array, columns=["time", "A", "B"])
    #print(train_x_df)
    kmeans = timeSeriesCluster.TimeSeriesCluster(timeSeriesCluster.METHOD_NAME_K_MEANS, max_n_clusters=10)

    kmeans.fit(train_x_df)
    #print(kmeans)

    test_x_array = np.array([[0, 0, 5], [1, 1, 5], [2, 2, 5], [3, 3, 5], [4, 4, 5]])
    test_x_df = pd.DataFrame(test_x_array, columns=["time", "A", "B"])
    new_test_df = kmeans.predict(test_x_df)
    print(new_test_df)
    # plot to select the best K for k-means
    cluster_utilities.plot_elbow_criterion(3, new_test_df, sk_KMeans)

    gaussianMixture = timeSeriesCluster.TimeSeriesCluster(
        timeSeriesCluster.METHOD_NAME_GAUSSIAN_MIXTURE_MODEL_AUTOCLUSTERING, max_n_clusters=3)
    gaussianMixture.fit(train_x_df)
    new_test_df = gaussianMixture.predict(test_x_df)
    print(new_test_df)
    # plot BIC scores
    gaussianMixture.model.plot_bic_scores()

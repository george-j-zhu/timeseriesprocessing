#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu

"""
time series clustering
"""
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn import preprocessing
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn import manifold
from sklearn.decomposition import KernelPCA

from .gaussianMixture import GaussianMixture
from .kMeans import KMeans
from . import cluster_utilities as clusterUtilities

METHOD_NAME_AFFINITY_PROPAGATION = "AffinityPropagation"
METHOD_NAME_AGGLOMERATIVE_CLUSTERING = "AgglomerativeClustering"
METHOD_NAME_BIRCH = "Birch"
METHOD_NAME_DBSCAN = "DBSCAN"
METHOD_NAME_GAUSSIAN_MIXTURE_MODEL = "GaussianMixture"
METHOD_NAME_K_MEANS = "k-means"
METHOD_NAME_MEAN_SHIFT = "MeanShift"
METHOD_NAME_SPECTRAL_CLUSTERING = "SpectralClustering"
METHOD_NAME_WARD = "Ward"

# TODO x-means is not available currently
METHOD_NAME_X_MEANS = "x-means"
METHOD_NAME_GAUSSIAN_MIXTURE_MODEL_AUTOCLUSTERING = "GaussianMixture_auto_clustering"


DIM_REDUCTION_METHOD_TSNE = "t-SNE"
DIM_REDUCTION_METHOD_KERNEL_PCA = "Kernel PCA"


class TimeSeriesCluster:
    """
    Cluster class for time series data set.
    number of clusters can not be decided automatically.

    Note:
    As DBSCAN does not support predict and fit_predict, use get_labels instead.
    """

    def __init__(self, cluster_method_name, normalize=False, max_n_clusters=10, **params):
        """
        constructor

        Args:
            n_cluster: number of cluster:
            params: this is a dictionary type that contains parameters for each clustering algorithm.
                    1.AffinityPropagation
                        damping: default: 0.5
                            Damping factor (between 0.5 and 1) is the extent to which the
                            current value is maintained relative to incoming values
                            (weighted 1 - damping). This in order to avoid numerical
                            oscillations when updating these values (messages).
                        preference: default: None
                            Preferences for each point - points with larger values of preferences
                            are more likely to be chosen as exemplars. The number of exemplars,
                            ie of clusters, is influenced by the input preferences value.
                            If the preferences are not passed as arguments, they will be set to
                            the median of the input similarities.
                    2.AgglomerativeClustering
                        n_neighbors: Number of neighbors for each sample.
                        n_clusters: The number of clusters to find.
                    3.Birch
                        n_clusters: The number of clusters to find.
                    4.DBSCAN
                        eps:
                    5.GaussianMixture
                        n_clusters: The number of clusters to find.
                    6.k-means
                        n_clusters: The number of clusters to find.
                    7.MeanShift
                        quantile:
                    8.SpectralClustering
                        n_clusters: The number of clusters to find.
                    9.Ward
                        n_neighbors: Number of neighbors for each sample.
                        n_clusters: The number of clusters to find.
        """
        np.random.seed(0)
        self.max_n_clusters = max_n_clusters
        self.params = params
        self.method_name = cluster_method_name
        # the following parameters will be initialized during fitting.
        self.model = None
        self.normalize = normalize
        self.scaler = None

        # TODO check necessary params for each algorithm.

    def __init_model(self, X):
        """
        initialize a model with the specified algorithm.

        Args:
            cluster_method_name: algorithm name
        """

        n_clusters = 2
        if "n_clusters" in self.params:
            n_clusters = self.params["n_clusters"]

        if self.method_name == METHOD_NAME_AFFINITY_PROPAGATION:
            damping = 0.5
            if "damping" in self.params:
                damping = self.params["damping"]
            preference = None
            if "preference" in self.params:
                preference = self.params["preference"]
            self.model = cluster.AffinityPropagation(damping=damping, preference=preference)
        elif self.method_name == METHOD_NAME_AGGLOMERATIVE_CLUSTERING:
            n_neighbors = 3
            if "n_neighbors" in self.params:
                n_neighbors = self.params["n_neighbors"]
            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(
                X, n_neighbors=n_neighbors, include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
            self.model = cluster.AgglomerativeClustering(
                linkage="average", affinity="cityblock",
                n_clusters=n_clusters, connectivity=connectivity)
        elif self.method_name == METHOD_NAME_BIRCH:
            self.model = cluster.Birch(n_clusters=n_clusters)
        elif self.method_name == METHOD_NAME_DBSCAN:
            eps = 0.5
            if "eps" in self.params:
                eps = self.params["eps"]
            min_samples = 3
            if "min_samples" in self.params:
                min_samples = self.params["min_samples"]
            self.model = cluster.DBSCAN(eps=eps, min_samples=min_samples)
        elif self.method_name == METHOD_NAME_GAUSSIAN_MIXTURE_MODEL:
            self.model = mixture.GaussianMixture(n_components=n_clusters)
        elif self.method_name == METHOD_NAME_GAUSSIAN_MIXTURE_MODEL_AUTOCLUSTERING:
            # Gaussian Mixture Model that decides n_components automatically by BIC scores.
            self.model = GaussianMixture(n_components=n_clusters,
                                         max_n_components=self.max_n_clusters)
        elif self.method_name == METHOD_NAME_K_MEANS:
            self.model = KMeans(n_clusters=n_clusters,
                                max_n_clusters=self.max_n_clusters)
        elif self.method_name == METHOD_NAME_MEAN_SHIFT:
            quantile = 3
            if "quantile" in self.params:
                n_neighbors = self.params["quantile"]
            bandwidth = cluster.estimate_bandwidth(X, quantile=quantile)
            self.model = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        elif self.method_name == METHOD_NAME_SPECTRAL_CLUSTERING:
            self.model = cluster.SpectralClustering(
                n_clusters=n_clusters, eigen_solver="arpack",
                affinity="nearest_neighbors")
        elif self.method_name == METHOD_NAME_WARD:
            n_neighbors = 3
            if "n_neighbors" in self.params:
                n_neighbors = self.params["n_neighbors"]
            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(
                X, n_neighbors=n_neighbors, include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
            self.model = cluster.AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward", connectivity=connectivity)
        else:
            print("algorithm not supported!")
            return

    def fit(self, train_x_with_time_df):
        """
        fit the model by the training data.

        Args:
            train_x_with_time_df: training data
            cluster_method_name: algorithm name (k-means as the default)
        """
        train_x_without_time_df = train_x_with_time_df[:]
        del train_x_without_time_df["time"]

        train_x_without_time_matrix = None
        if self.normalize is True:
            self.scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-0.5, 0.5))
            train_x_without_time_matrix = self.scaler.fit_transform(
                train_x_without_time_df)
        else:
            train_x_without_time_matrix = train_x_without_time_df.values

        self.__init_model(train_x_without_time_matrix)
        self.model.fit(train_x_without_time_matrix)

    def predict(self, test_x_with_time_df, plot_clusters=False,
                dim_reduction_method=DIM_REDUCTION_METHOD_TSNE, perplexity=30):
        """
        make predictions.
        as algorithms that are not based on centroid methods have no centroids, it makes sense to reclustering.

        Args:
            test_x_with_time_df: test data
        """
        # in order to retain the original object, use slice to do a shallow copy.
        test_x_without_time_df = test_x_with_time_df.copy()
        ret = test_x_with_time_df.copy()
        del test_x_without_time_df["time"]

        if hasattr(self.model, 'predict'):
            if self.normalize is True:
                test_x_without_time_matrix = self.scaler.transform(test_x_without_time_df)
            else:
                test_x_without_time_matrix = test_x_without_time_df.values
            predictions_array = self.model.predict(test_x_without_time_matrix)
        else:
            raise ValueError("Select clustering method does not support predicting.")

        if plot_clusters is True:
            self.__plot_clusters_onto_2D(self.model, test_x_without_time_matrix,
                                         dim_reduction_method=dim_reduction_method,
                                         perplexity=perplexity, plot=plot_clusters)

        # combine predictions_array, self.test_x_with_time_df as a new dataframe.
        ret["cluster_id"] = predictions_array
        return ret

    def fit_predict(self, train_x_with_time_df, plot_clusters=False,
                    dim_reduction_method=DIM_REDUCTION_METHOD_TSNE, perplexity=30):
        """
        initialize a model with the specified algorithm.
        fit the model by the training data.
        make predictions by the training data.

        Args:
            train_x_with_time_df: training data
            cluster_method_name: algorithm name (k-means as the default)
        """
        train_x_without_time_df = train_x_with_time_df.copy()
        ret = train_x_with_time_df.copy()
        del train_x_without_time_df["time"]

        train_x_without_time_matrix = None
        if self.normalize is True:
            self.scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-0.5, 0.5))
            train_x_without_time_matrix = self.scaler.fit_transform(
                train_x_without_time_df)
        else:
            train_x_without_time_matrix = train_x_without_time_df.values

        self.__init_model(train_x_without_time_matrix)
        self.model.fit(train_x_without_time_matrix)

        if hasattr(self.model, 'predict'):
            predictions_array = self.model.predict(train_x_without_time_matrix)
        else:
            raise ValueError("Select clustering method does not support predicting.")

        if plot_clusters is True:
            self.__plot_clusters_onto_2D(self.model, train_x_without_time_matrix,
                                         dim_reduction_method=dim_reduction_method,
                                         perplexity=perplexity, plot=plot_clusters)
        ret["cluster_id"] = predictions_array
        return ret

    def get_params(self, deep=True):
        """
        get parameters of this model
        """
        return self.model.get_params(deep)

    def get_labels(self):
        """
        get labels for all learning samples.
        """
        return self.model.labels_

    def get_n_clusters(self):
        """
        get n_clusters
        """
        return clusterUtilities.get_n_clusters(self.model)

    def select_n_clusters_with_silhouette(self, data, plot=False,
                                          dim_reduction_method=DIM_REDUCTION_METHOD_TSNE,
                                          perplexity=30):
        """
        Selecting the number of clusters with silhouette analysis on clustering.

        From Wikipedia:
        Silhouette refers to a method of interpretation and validation of consistency within
        clusters of data. The technique provides a succinct graphical representation of how well
        each object lies within its cluster.
        The silhouette value is a measure of how similar an object is to its own cluster
        (cohesion) compared to other clusters (separation). The silhouette ranges from −1 to +1,
        where a high value indicates that the object is well matched to its own cluster(far away
        from the neighboring clusters) and poorly matched to neighboring clusters. If most objects
        have a high value, then the clustering configuration is appropriate. If many points have a
        low or negative value, then the clustering configuration may have too many or too few
        clusters.
        The silhouette can be calculated with any distance metric, such as the Euclidean distance
        or the Manhattan distance.

        bad group for the given data are usually due to the presence of clusters with below
        average silhouette scores and also due to wide fluctuations in the size of the silhouette
        plots（a major of the samples are grouped into some clusters while other cluster contains
        only few sample）.
        """

        # copy the input arg X to retained the original object
        input_df = data.copy()
        del input_df["time"]

        X = None
        if self.normalize is True:
            X = pd.DataFrame(preprocessing.StandardScaler().fit_transform(input_df),
                             index=input_df.index, columns=input_df.columns)
        else:
            X = input_df

        for n_clusters in range(2, self.max_n_clusters):

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = None

            if self.method_name == METHOD_NAME_AGGLOMERATIVE_CLUSTERING:
                # connectivity matrix for structured Ward
                connectivity = kneighbors_graph(
                    X, n_neighbors=self.params["n_neighbors"], include_self=False)
                # make connectivity symmetric
                connectivity = 0.5 * (connectivity + connectivity.T)
                clusterer = cluster.AgglomerativeClustering(
                    linkage="average", affinity="cityblock",
                    n_clusters=n_clusters, connectivity=connectivity)
            elif self.method_name == METHOD_NAME_BIRCH:
                clusterer = cluster.Birch(n_clusters=n_clusters)
            elif self.method_name == METHOD_NAME_GAUSSIAN_MIXTURE_MODEL:
                clusterer = mixture.GaussianMixture(
                    n_components=n_clusters, covariance_type="full")
            elif self.method_name == METHOD_NAME_K_MEANS:
                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            elif self.method_name == METHOD_NAME_SPECTRAL_CLUSTERING:
                clusterer = cluster.SpectralClustering(
                    n_clusters=n_clusters, eigen_solver="arpack",
                    affinity="nearest_neighbors")
            elif self.method_name == METHOD_NAME_WARD:
                # connectivity matrix for structured Ward
                connectivity = kneighbors_graph(
                    X, n_neighbors=self.params["n_neighbors"], include_self=False)
                # make connectivity symmetric
                connectivity = 0.5 * (connectivity + connectivity.T)
                clusterer = cluster.AgglomerativeClustering(
                    n_clusters=n_clusters, linkage="ward", connectivity=connectivity)
            else:
                print("silhouette analysis can not be applied to algorithms that automatically select n_clusters.")
                print("algorithm name: " + self.method_name)
                return

            if self.normalize is True:
                self.scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-0.5, 0.5))
                X_matrix = self.scaler.fit_transform(X)
                clusterer.fit(self.scaler.fit_transform(X))
            else:
                X_matrix = X.values

            clusterer.fit(X_matrix)
            self.__plot_clusters_onto_2D(clusterer, X_matrix, dim_reduction_method=dim_reduction_method,
                                         perplexity=perplexity, plot=plot)


    def plot_clusters_onto_2D(self, data_df, dim_reduction_method="t-SNE",
                              perplexity=30, plot=False):
        """
        For high dimensional data, use t-SNE to reduce the dimensionality and plot result
        on a 2D plane.
        Args:
            perplexity: The perplexity is related to the number of nearest neighbors that is used
                        in other manifold learning algorithms. Larger datasets usually require a
                        larger perplexity. Consider selecting a value between 5 and 50. The choice
                        is not extremely critical since t-SNE is quite insensitive to this parameter.
        """
        data_without_time_df = data_df.copy()
        del data_without_time_df["time"]

        data_without_time_matrix = None
        if self.normalize is True:
            if self.scaler is None:
                raise ValueError("Call fit or fit_predict first!")

            data_without_time_matrix = self.scaler.fit_transform(data_without_time_df)
        else:
            data_without_time_matrix = data_without_time_df.values

        self.__plot_clusters_onto_2D(self.model, data_without_time_matrix, dim_reduction_method,
                                     perplexity=perplexity, plot=plot)

    def __plot_clusters_onto_2D(self, clusterer, X, dim_reduction_method,
                                perplexity, plot=False):
        """
        For high dimensional data, use t-SNE to reduce the dimensionality and plot result
        on a 2D plane.
        Args:
            perplexity: The perplexity is related to the number of nearest neighbors that is used
                        in other manifold learning algorithms. Larger datasets usually require a
                        larger perplexity. Consider selecting a value between 5 and 50. The choice
                        is not extremely critical since t-SNE is quite insensitive to this parameter.
        """

        if hasattr(clusterer, 'predict'):
            cluster_labels = clusterer.predict(X)
        else:
            cluster_labels = clusterer.labels_
        if len(set(cluster_labels)) == 1:
            print("clustering failed. unable to group data into two more clusters.")
            return

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)

        n_clusters = clusterUtilities.get_n_clusters(clusterer)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        if plot is True:
            cmap = cm.get_cmap("CMRmap")
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                                                sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                # cm.spectral has been removed since matplotlib 2.2
                #color = cm.spectral(float(i) / n_clusters)
                color = cmap(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # map X (high dimensional) to 2D
            if X.shape[1] > 2:
                # reduce the dimensionality of input data to 2D
                if dim_reduction_method == DIM_REDUCTION_METHOD_KERNEL_PCA:
                    kpca = KernelPCA(n_components=2, kernel="rbf",
                                     gamma=10, random_state=0)
                    X = kpca.fit_transform(X)
                else:
                    tsne = manifold.TSNE(n_components=2, perplexity=perplexity,
                                         init='pca', random_state=0)
                    X = tsne.fit_transform(X)

            # 2nd Plot showing the actual clusters formed
            colors = cmap(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=100, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            ax2.set_title("The visualization of the clustered data({}).".format(dim_reduction_method))
            ax2.set_xlabel("reduced feature space of 1st dimension")
            ax2.set_ylabel("reduced feature space of 2nd dimension")

            plt.suptitle(("Silhouette analysis for clustering methods "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

            plt.show()

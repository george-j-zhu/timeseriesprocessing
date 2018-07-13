import unittest

from timeseriesprocessing.featureengineering.feature_selection import FeatureSelector
import pandas as pd
from sklearn.datasets import make_classification, load_iris, make_blobs
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_regression

class TestFeatureSelector(unittest.TestCase):
    """
    test class of FeatureSelector

    test list:
        select_high_variance_features
        select_features_by_feature_scores
        get_feature_coefficients
        get_feature_importances
        get_feature_ranking_by_recursively_eliminating_features
        get_new_features_by_clustering_features
    """

    def test_select_high_variance_features(self):
        """
        test select_high_variance_features
        """

        data_matrix = [[0.2, 1.5, 2.6], [2.8, 3.6, 2.5], [1.6, 2.0, 0.1],
                       [1.0, 2.5, 3.2], [1.1, 8.0, 6.1]]
        X_df = pd.DataFrame(data_matrix, columns=["X1", "X2", "X3"])
        y_df = pd.DataFrame([[8.0], [9.0], [10.0], [11.5], [12.1]], columns=["Y"])

        expected = X_df.loc[:, ["X2", "X3"]]

        fs = FeatureSelector(X_df, y_df)
        actual = fs.select_high_variance_features(threshold=1.0)
        self.assertEqual(expected.columns[0], actual.columns[0])
        self.assertEqual(expected.columns[1], actual.columns[1])

    def test_select_features_by_feature_scores(self):
        """
        test select_features_by_feature_scores
        """

        data_matrix = [[0.2, 1.5, 2.6], [2.8, 3.6, 2.5], [1.6, 2.0, 0.1],
                       [1.0, 2.5, 3.2], [1.1, 8.0, 6.1]]
        X_df = pd.DataFrame(data_matrix, columns=["X1", "X2", "X3"])
        y_df = pd.DataFrame([[8.0], [9.0], [10.0], [11.5], [12.1]], columns=["Y"])

        expected = X_df.loc[:, ["X2", "X3"]]

        fs = FeatureSelector(X_df, y_df)
        actual = fs.select_features_by_feature_scores(n_features=2)
        self.assertEqual(expected.columns[0], actual.columns[0])
        self.assertEqual(expected.columns[1], actual.columns[1])

    def test_get_feature_coefficients(self):
        """
        test select_features_by_linear_model
        """
        X, y, _ = make_regression(n_samples=10000,
                                     n_features=100, noise=0.1, coef=True)

        fs = FeatureSelector(pd.DataFrame(X), pd.DataFrame(y))
        actual = fs.get_feature_coefficients(norm_prior=0)
        print(actual)
        #self.assertEqual(0.43034753526246899, actual[1])

    def test_get_feature_importances(self):
        """
        test get_feature_importances
        """
        X, y, coef = make_regression(n_samples=10000,
                                     n_features=100, noise=0.1, coef=True)

        fs = FeatureSelector(pd.DataFrame(X), pd.DataFrame(y))
        actual = fs.get_feature_importances(n_estimators=10)
        print(actual)
        #self.assertEqual(0.43034753526246899, actual[1])

    def test_get_feature_ranking_by_recursively_eliminating_features(self):
        """
        test get_feature_ranking_by_recursively_eliminating_features
        """
        # Build a classification task using 3 informative features
        X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                                   n_redundant=2, n_repeated=0, n_classes=8,
                                   n_clusters_per_class=1, random_state=0)
        # Create the RFE object and compute a cross-validated score.
        svc = SVC(kernel="linear")
        fs = FeatureSelector(pd.DataFrame(X), pd.DataFrame(y))
        actual_ranking = fs.get_feature_ranking_by_recursively_eliminating_features(svc, cv=2)
        print(actual_ranking)

    def test_get_new_features_by_clustering_features(self):
        """
        test get_new_features_by_clustering_features
        """

        # Generating the sample data from make_blobs
        # This particular setting has one distinct cluster and 3 clusters placed close
        # together.
        X, y = make_blobs(n_samples=500,
                          n_features=10,
                          centers=4,
                          cluster_std=1,
                          center_box=(-10.0, 10.0),
                          shuffle=True,
                          random_state=1)  # For reproducibility

        fs = FeatureSelector(pd.DataFrame(X), pd.DataFrame(y))
        centers = fs.get_new_features_by_clustering_features(4)
        self.assertEqual(centers.shape[1], 4)


if __name__ == "__main__":
    unittest.main()

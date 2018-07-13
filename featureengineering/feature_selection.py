#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu

import gc
import warnings
from scipy.sparse import issparse
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso, Ridge, OrthogonalMatchingPursuitCV
from sklearn.cluster import KMeans as sk_KMeans

class FeatureSelector():
    """
    FeatureSelector class provides some convenient method to do feature selection.

    Implements some different methods to find features machine learning.
        1. select_high_variance_features
               find all features whose variance meet the threshold.
        2. select_features_by_feature_scores
               find all feature by feature scores which are calculated by a score function.
        3. get_feature_coefficients
               get feature coefficients by fitting a linear model. norm prior can be specified to
               identify more or less important features.
        4. get_feature_importances
               get feature importance using xgboost with cross-validation. This is a different
               approach from a linear model.
        5. get_feature_ranking_by_recursively_eliminating_features
               A recursive feature elimination method to find the best n features.
        6. get_new_features_by_clustering_features
               create new features by clustering.
               This is not a feature selection method.

    Here are some tips about how to do feature selection:
    refer to paper [An Introduction to Variable and Feature Selection]
    from https://dl.acm.org/citation.cfm?id=944968

    Abstract:
    1.Do you have domain knowledge?
        If yes, construct a better set of “ad hoc” features.
    2.Are your features commensurate?
        If no, consider normalizing them.
    3.Do you suspect interdependence of features?
        If yes, expand your feature set by constructing conjunctive features or products of
        features, as much as your computer resources allow you.
    4.Do you need to prune the input variables(e.g.  for cost, speed or data understanding reasons)?
        If no, construct disjunctive features or weighted sums of features (e.g. by clustering or
        matrix factorization).
    5.Do you need to assess features individually(e.g. to understand their influence on the system
      or because their number is so large that you need to do a first filtering)?
        If yes, use a variable ranking method; else, do it anyway to get baseline results.
    6.Do you need a predictor?
        If no, stop.
    7.Do you suspect your data is “dirty”(has a few meaningless input patterns and/or noisy outputs
      or wrong class labels)?
        If yes, detect the outlier examples using the top ranking variables obtained in step 5 as
        representation; check and/or discard them.
    8.Do you know what to try first?
        If no, use a linear predictor. Use a forward selection method with the “probe” method as a
        stopping criterion (Section 6) or use L0-norm(find the sparsest solution) embedded method
        (Section 4.3).  For comparison, following the ranking of step 5, construct a sequence of
        predictors of same nature using increasing subsets of features. Can you match or improve
        performance with a smaller subset?
        If yes, try a non-linear predictor with that subset.
    9.Do you have new ideas, time, computational resources, and enough examples?
        If yes, compare several feature selection methods, including your new idea, correlation
        coefficients, backward selection and embedded methods (Section 4). Use linear and non-linear
        predictors.Select the best approach with model selection (Section 6).
    10.Do you want a stable solution(to improve performance and/or understanding)?
        If yes, sub-sample your data and redo your analysis for several “bootstraps” (Section 7.1).
    """

    def __init__(self, X_df, y_df=None):
        """
        constructor.

        Args:
            X_df: features
            y_df: labels
        """
        self.X_df = X_df
        self.y_df = y_df
        self.feature_variances_array = None
        self.feature_scores_array = None
        self.feature_ranking_array = None
        self.coef_ = None
        self.feature_importance_ = None

    def select_high_variance_features(self, threshold=0.0):
        """
        removes all features whose variance doesn’t meet the threshold. By default, it removes all
        zero-variance features, i.e. features that have the same value in all samples.

        Args:
            threshold:
        """
        if self.feature_variances_array is None:
            sel = VarianceThreshold(threshold=threshold)
            sel.fit(self.X_df.values)
            self.feature_variances_array = sel.variances_

        ret_df = self.X_df.copy()
        for col_name, variance in zip(self.X_df.columns, self.feature_variances_array):
            if variance <= threshold:
                # variance is less than or equal to threshold
                ret_df = ret_df.drop(col_name, axis=1)

        return ret_df

    def select_features_by_feature_scores(self, score_func=mutual_info_regression, n_features=1):
        """
        select the best features based on univariate statistical tests.
        It can be seen as a preprocessing step to an estimator.

        F-test score functions like mutual_info_regression are used to return univariate scores
        and p-values.
        The F-test methods estimate the degree of linear dependency between two random variables.
        Args:
            score_func: Function taking two arrays X and y, and returning a pair of arrays
                        (scores, pvalues) or a single array with scores. Default is
                        mutual_info_regression
            n_features: n features to select from X_df
        """

        if self.y_df is None:
            raise ValueError("No training labels provided.")

        selector = None
        if issparse(self.X_df.values) and score_func == f_regression:
            # F-test is used to calculate linear correlations.
            warnings.warn("F-test on a sparse matrix is not recommended.")
        selector = SelectKBest(score_func, k='all')
        selector.fit(self.X_df, self.y_df)
        # use scores_ or pvalues_
        self.feature_scores_array = selector.scores_

        # get top n_features by scores
        scores = {}
        for idx, score in enumerate(selector.scores_):
            scores[self.X_df.columns[idx]] = score

        scores = sorted(scores.items(), reverse=True, key=lambda x: x[1])

        print("sorted scores for each feature:", scores)

        ret = pd.DataFrame()
        for idx in range(n_features):
            ret[scores[idx][0]] = self.X_df.loc[:, scores[idx][0]]
        # identify which features are selected
        return ret

    def get_feature_coefficients(self, norm_prior=1):
        """
        get feature coefficients using linear regression.
        Linear models penalized with the L1 norm have sparse solutions: many of their estimated
        coefficients are zero.
        Args:
            norm_prior: 1 for L1-norm as default. use L0 to get the sparsest result.
        """
        model = None
        alphas = np.logspace(-4, -0.5, 30)
        tuned_parameters = [{'alpha': alphas}]
        coefficient_value = None
        if norm_prior == 0:
            # L0-norm
            model = OrthogonalMatchingPursuitCV()
            model.fit(self.X_df.values, self.y_df.values)
            coefficient_value = model.coef_
        elif norm_prior == 1:
            # L1-norm
            # Lasso
            lasso = Lasso(random_state=0)
            n_folds = 3
            gridsearch = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
            gridsearch.fit(self.X_df.values, self.y_df.values)
            coefficient_value = gridsearch.best_estimator_.coef_
        elif norm_prior == 2:
            # L2-norm
            # Ridge
            ridge = Ridge(random_state=0)
            n_folds = 3
            gridsearch = GridSearchCV(ridge, tuned_parameters, cv=n_folds, refit=False)
            gridsearch.fit(self.X_df.values, self.y_df.values)
            coefficient_value = gridsearch.best_estimator_.coef_
        else:
            print("invalid norm!")

        self.coef_ = coefficient_value
        return coefficient_value

    def get_feature_importances(self, eval_metric="mae", n_estimators=10,
                                n_iterations=10, early_stopping = True):
        """
        get feature importance according to a gradient boosting machine.
        The xgboost can be trained with early stopping using a validation set to prevent
        over-fitting. The feature importances are averaged over `n_iterations` to reduce variance.

        Args:
            eval_metric : evaluation metric to use for the gradient boosting machine for early
                          stopping. Must be provided if `early_stopping` is True
            n_estimators: number of trees, change it to 1000 for better results
            n_iterations : Number of iterations to train the gradient boosting machine
            early_stopping : Whether or not to use early stopping with a validation set when
                             training
        """

        if early_stopping and eval_metric is None:
            raise ValueError("""eval metric must be provided with early stopping.""")

        if self.y_df is None:
            raise ValueError("No training labels provided.")

        # Extract feature names
        feature_names = list(self.X_df.columns)

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))

        print('Training Gradient Boosting Model\n')

        # Iterate through each fold
        for _ in range(n_iterations):

            model = XGBRegressor()
            gridsearch = None
            # brute force scan for all parameters, here are the tricks
            # usually max_depth is 6,7,8
            # learning rate is around 0.05, but small changes may make big diff
            # tuning min_child_weight subsample colsample_bytree can have
            # much fun of fighting against overfit
            # n_estimators is how many round of boosting
            # finally, ensemble xgboost with multiple seeds may reduce variance
            params_grid = {'n_jobs':[4], #when use hyperthread, xgboost may become slower
                           'learning_rate': [0.05, 0.1], #so called `eta` value
                           'max_depth': [6, 7, 8],
                           'min_child_weight': [11],
                           'silent': [1],
                           'subsample': [0.8, 0.85, 0.9, 0.95],
                           'colsample_bytree': [0.5, 1.0],
                           'n_estimators': [n_estimators],
                           'random_state': [1337]}

            # If training using early stopping need a validation set
            if early_stopping:
                train_x, test_x, train_y, test_y = train_test_split(self.X_df.values,
                                                                    self.y_df.values,
                                                                    test_size = 0.15)

                fit_params = {"early_stopping_rounds":100, "eval_metric" : eval_metric,
                              "eval_set" : [[test_x, test_y]]}
                # Train the model with early stopping
                gridsearch = GridSearchCV(model, params_grid, verbose=0, fit_params=fit_params,
                                          cv=10, scoring="neg_mean_squared_error")
                gridsearch.fit(train_x, train_y)
                # Clean up memory
                gc.enable()
                del train_x, train_y, test_x, test_y
                gc.collect()
            else:
                gridsearch = GridSearchCV(model, params_grid, verbose=0,
                                          cv=10, scoring="neg_mean_squared_error")
                gridsearch.fit(self.X_df.values, self.y_df.values)

            # Record the feature importances
            feature_importance_values += gridsearch.best_estimator_.feature_importances_ / n_iterations

        self.feature_importance_ = feature_importance_values

        return feature_importance_values


    def get_feature_ranking_by_recursively_eliminating_features(self, estimator,
                                                                cv=None, n_features=1):
        """
        A recursive feature elimination method to find the best n features.

        Args:
            estimator: regressor
            cv: cross-validation generator or an iterable. automatic tuning of the number of
                features selected with cross-validation.
            n_features: n features to select. only effective when cv is not specified.
        """
        rfe = None
        if cv is None:
            rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
            rfe.fit(self.X_df.values, self.y_df.values)
        else:
            rfe = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(cv),
                        scoring='accuracy')
            rfe.fit(self.X_df.values, self.y_df.values)
            print("Optimal number of features : %d" % rfe.n_features_)

        self.feature_ranking_array = rfe.ranking_
        return rfe.ranking_

    def get_new_features_by_clustering_features(self, n_clusters):
        """
        use k-means to group features. As features in each group can be considered similar,
        replace a group of “similar” features by a cluster centroid,  which becomes a new feature.
        """
        clusterer = sk_KMeans(n_clusters=n_clusters).fit(self.X_df.values.T)
        return clusterer.cluster_centers_.T

    def extract_features(self, dataframe, in_dim, out_dim):
        """
        use neural network forward propagation to extract new features.
        TODO
        """
        pass

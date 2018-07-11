#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu

"""
utilites
"""

import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import datetime
from sklearn import preprocessing, linear_model
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from . import constants


def get_dimension(array_list):
    """
    get dimension of an array.

    Args:
        array_list: a numpy.array instance

    Returns:
        array contains number of rows and columns
    """
    return len(np.shape(array_list))


def __initialize_model(model_name, lamda=0, hyper_parameters={}):
    """
    initialize machine learning model.

    Args:
        model_name: learning algorithm name
        lamda: coefficient of standardization item
        hyper_parameter: other parameters for algorithms
               See parameters for RandomForest Regression in sci-kit-learn

    Returns:
        an initialized classifier
    """
    if model_name == constants.MODEL_NAME_LASSO:
        # note: alpha in scikit-learn reprsents lamda which is the constant that
        # multiplies the regularization term
        clf_lasso = linear_model.Lasso(alpha=lamda)
        return clf_lasso
    elif model_name == constants.MODEL_NAME_ELASTICNET:
        clf_elasticnet = ElasticNet(alpha=lamda)
        return clf_elasticnet
    elif model_name == constants.MODEL_NAME_RIDGE:
        clf_ridge = linear_model.Ridge(alpha=lamda)
        return clf_ridge
    elif model_name == constants.MODEL_NAME_RIDGECV:
        clf_ridgecv = linear_model.RidgeCV(alphas=constants.lamdaArray)
        return clf_ridgecv
    elif model_name == constants.MODEL_NAME_LARS:
        clf_lars = linear_model.Lars(n_nonzero_coefs=1)
        return clf_lars
    elif model_name == constants.MODEL_NAME_BAYESIAN:
        clf_bayesian = linear_model.BayesianRidge()
        return clf_bayesian
    elif model_name == constants.MODEL_NAME_SGD:
        clf_sgd = linear_model.SGDRegressor(alpha=lamda)
        return clf_sgd
    elif model_name == constants.MODEL_NAME_RANDOM_FOREST:
        clf_random_forest = RandomForestRegressor(**hyper_parameters, random_state=0, n_jobs=-1)
        return clf_random_forest


def build_model(model_name, X_train_matrix, Y_train_matrix, loop_lamda=True):
    """
    construct a learning model using the specified algorithm.

    Args:
        model_name: learning algorithm name
        X_train_matrix: training data of explainatory variables
        Y_train_matix: training data of predictor variable
        loop_lamda: True for using built-in CV to select the best lamda

    Returns:
        learning model using the specifiled learning algorithm
    """
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    if model_name == constants.MODEL_NAME_RANDOM_FOREST:
        r2_scoring = make_scorer(r2_score)
        clf_model_untuned = __initialize_model(model_name)
        # use gridsearchcv to find the best parameters for ramdon forest.
        forest_grid_search = GridSearchCV(clf_model_untuned, constants.RAMDON_FOREST_PARAMS,
                                          scoring=r2_scoring, cv=cv)
        forest_grid_search.fit(X_train_matrix, Y_train_matrix.transpose()[0])
        # initialize a new model with the best parameters.
        # print("parameters for random forest: {0}".format(forest_grid_search.best_params_), sep="\n")
        clf_model = __initialize_model(model_name, forest_grid_search.best_params_)
    else:
        if loop_lamda:
            best_lamda = 0.0
            prev_score = 0.0
            for lamda in constants.lamdaArray:
                clf_model = __initialize_model(model_name, lamda)
                clf_model.fit(X_train_matrix, Y_train_matrix.transpose()[0])
                # use cross validation to get the best lamda
                score = np.mean(cross_val_score(clf_model, X_train_matrix, Y_train_matrix, cv=cv))
                if prev_score < score:
                    best_lamda = lamda
                    prev_score = score
            # build model with the best lamda
            clf_model = __initialize_model(model_name, best_lamda)
        else:
            clf_model = __initialize_model(model_name)

    clf_model.fit(X_train_matrix, Y_train_matrix.transpose()[0])

    # plot learning curves
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plt.figure(figsize=(30, 8))
    plt.subplot(1, 3, 1)
    title = "Learning Curves(" + model_name + ")"
    estimator = __initialize_model(model_name, 0.3)
    plot_learning_curve(estimator, title, X_train_matrix, Y_train_matrix.transpose()[0],
                        ylim=(0.3, 1.01), cv=cv, n_jobs=4)
    plt.show()

    return clf_model


def predict_future_data(clf_model, X_test_matrix):
    """
    make predicitions.

    Args:
        clf_model: learning model
        X_test_matrix: test dataset of explainatory variables

    Returns:
        predictions
    """
    # predictions
    predictions = clf_model.predict(X_test_matrix)
    if get_dimension(predictions) == 1:
        predictions = np.transpose(np.array([predictions]))
    # mean absolute error
    #MAE = metrics.mean_absolute_error(Y_test_matrix_dump_amount, predictions_dump_amount)
    #MAE = float("{0:.1f}".format(MAE))
    #print(MAE)

    return predictions


def evaluate_model(model_name, clf_model, X_test_matrix, Y_test_matrix, columns, title):
    """
    evaluate the model and plot the coefficient of each explainatory variable

    Args:
        model_name: algorithm name
        clf_model: learning model
        X_test_matrix: test dataset of explainatory variables
        Y_test_matrix: test dataset of predicitor variables
        columns: column of explainatory variables
        title: title of the graph for showing predicted/real values

    Returns:
        predictions
    """
    if hasattr(clf_model, "coef_"):
        # print("重み: {0}".format(clf_model.coef_), sep="\n")
        importances = clf_model.coef_
    else:
        # print("重み: {0}".format(clf_model.feature_importances_), sep="\n")
        importances = clf_model.feature_importances_
    # plot weights
    if 1 == (importances.shape)[0]:
        importances = importances[0]
    indices = np.argsort(importances)
    plt.figure(figsize=(30, 8))
    plt.subplot(1, 3, 1)
    plt.title("coefficient", **constants.sfont)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), np.array(list(columns))[indices], **constants.sfont)
    plt.xticks(fontsize=18)

    # predictions
    predictions = predict_future_data(clf_model, X_test_matrix)
    # mean absolute error
    #print(Y_test_matrix)
    #print(predictions)
    mae = metrics.mean_absolute_error(Y_test_matrix, predictions)

    plt.subplot(1, 3, 2)
    plt.title(title + " Accurracy", **constants.sfont)
    plt.scatter(Y_test_matrix, predictions, label="Predicted")
    if Y_test_matrix.max() > predictions.max():
        plotmax = Y_test_matrix.max()
    else:
        plotmax = predictions.max()
    x = np.linspace(0, plotmax)
    y = x
    plt.plot(x, y, "r-")      # draw a diagonal line
    #accuracy = round(metrics.explained_variance_score(Y_test_matrix, predictions) * 100, 2)
    accuracy = round(metrics.r2_score(Y_test_matrix, predictions) * 100, 2)
    legend = plt.legend(loc=2, title="[{0}:{1:.2f}%]".format(model_name, accuracy), prop=constants.fp)
    plt.setp(legend.get_title(), fontsize='20')
    plt.xlabel("real values", **constants.sfont)
    plt.ylabel("predicted values", **constants.sfont)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plot learning curves
    #cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    #plt.subplot(1,3,3)
    #title = "Learning Curves(" + model_name + ")"
    #estimator = __initialize_model(model_name, 0.3)
    #plot_learning_curve(estimator, title, X_test_matrix, Y_test_matrix.transpose()[0],
    #                    ylim=(0.3, 1.01), cv=cv, n_jobs=4)
    plt.show()

    return predictions, accuracy, mae


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    from scikit-learn document.
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def impute_missing_data(dataframe):
    """
    replace NaN value with the mean value.

    Args:
        data_matrix: input data

    Returns:
        imputed data matrix
    """
    imp = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0)
    return pd.DataFrame(imp.fit_transform(dataframe), index=dataframe.index,
                        columns=dataframe.columns)


def scale_data(dataframe):
    """
    feature scaling (mean normalization)

    Args:
        dataframe: input data

    Returns:
        mean normalized dataframe
    """
    return pd.DataFrame(preprocessing.StandardScaler().fit_transform(dataframe), index=dataframe.index,
                        columns=dataframe.columns)


def range_scale_data(dataframe):
    """
    scale data to a specified range

    Args:
        dataframe: input data

    Returns:
        range scaled dataframe
    """
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-0.5, 0.5))
    return pd.DataFrame(min_max_scaler.fit_transform(dataframe), index=dataframe.index,
                        columns=dataframe.columns)


def quantile_transform_data(data_matrix):
    """
    puts each feature into the same range or distribution.
    it smooths out unusual distributions but distort correlations and distances within
    and across features

    Args:
        data_matrix: input data

    Returns:
        quantiled matrix
    """
    quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
    return quantile_transformer.fit_transform(data_matrix)


def add_polynomial_features(data_matrix):
    """
    transform the features to polynominal terms
    eg) when degree=2 the features from (X1, X2) to (1, X1, X2, X1^2, X1*X2, X2^2).

    Args:
        data_matrix: input data

    Returns:
        matrix with polynomial terms
    """
    poly = preprocessing.PolynomialFeatures(2)
    return poly.fit_transform(data_matrix)


def plot_predictions(time_df, actual_data, predicted_data, start_datetime,
                     end_datetime, graph_name, locator=dates.HourLocator(interval=1)):
    """
    plot the predicted values with actual values for comparision.

    Args:
        time_df: time data
        actual_data: real values
        predicted_data: predicted values
        start_datetime: start datetime for plotting
        end_datetime: end datetime for plotting
        graph_name: graph name
        locator: tick locator

    Returns:
        None
    """

    # format datetime into YYYY/MM/DD HH:MM
    start_pos = 0
    end_pos = 0

    start_datetime_dt = datetime.strptime(start_datetime, "%Y/%m/%d %H:%M")
    end_datetime_dt = datetime.strptime(end_datetime, "%Y/%m/%d %H:%M")
    for index, row in time_df.iterrows():
        temp_datetime_dt = datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S")
        if temp_datetime_dt == end_datetime_dt:
            end_pos = index
        if temp_datetime_dt == start_datetime_dt:
            start_pos = index

    if end_pos > 0:
        # plot data
        time_matrix = dates.date2num(
            [datetime.strptime(t, "%Y/%m/%d %H:%M:%S") for t in time_df.as_matrix().transpose()[0]])
        fig, ax1 = plt.subplots(figsize=(len(time_df.index)/3, 12))
        ax1.plot(time_matrix[start_pos:end_pos], actual_data[start_pos:end_pos],
                 label=graph_name + "(real)",
                 color=constants.colors["violet"])
        ax1.plot(time_matrix[start_pos:end_pos], predicted_data[start_pos:end_pos],
                 label=graph_name + "(predicted)",
                 color='C1')
        ax1.legend(loc=2, prop=constants.fp)
        ax1.set_xticklabels(time_matrix[start_pos:end_pos], rotation=90)
        ax1.xaxis.set_major_locator(locator)
        hfmt = dates.DateFormatter("%Y/%m/%d %H:%M")
        ax1.xaxis.set_major_formatter(hfmt)
        ax1.tick_params(labelsize=18)
        plt.show()
    else:
        print("no data to plot!")


def export_predictions_to_csv(time_matrix, predicted_data, prediction_name, algorithm_name, export_dir):
    """
    export predicted values to csv files.

    Args:
        time_matrix: time data
        predicted_data: predicted values
        prediction_name: name for prediction
        algorithm_name: algorithm name
        export_dir: directory for exporting as csv files

    Returns:
        None
    """
    predictions_matrix = np.concatenate([time_matrix, predicted_data], axis=1)
    predictions_df = pd.DataFrame(predictions_matrix, columns=["time", "predicted_values"])
    predictions_df.to_csv(export_dir + prediction_name + "prediction_results(" + algorithm_name + ").csv",
                          index=False)
    print("predictions have been exported and saved")
    print(export_dir + prediction_name + "prediction_results(" + algorithm_name + ").csv")
    print("\n")


def plot_data(time_df, x_df, x_columns, x_colors, y_df, y_columns, y_colors, start_datetime,
              end_datetime, title, limit_vertical_maxval=False, save_figure=False):
    """
    plot data

    Args:
        time_matrix: time data
        x_df: dataframe for explanatory variables
        x_columns: columns for explanatory variables
        x_colors: plot colors for explanatory variables
        y_df: dataframe for target variables
        y_columns: columns for target variables
        y_colors: plot colors for target variables
        start_datetime: start datetime for plotting
        end_datetime: end datetime for plotting
        title: graph title
        limit_vertical_maxval: vertical axis max limitation
        save_figure: save graphs as files

    Returns:
        None
    """

    # TODO make days_per_plot as an argument of this method
    days_per_plot = 4

    if save_figure is True:
        start_pos = 0
        for i in range(len(time_df.index)):
            if i % (24 * days_per_plot) == 0 and i != 0:
                start_pos = i - 24 * days_per_plot
                end_pos = i
                fig = create_plot(time_df, x_df, x_columns, x_colors,
                                  title, limit_vertical_maxval, start_pos, end_pos,
                                  y_df, y_columns, y_colors)
                plt.savefig("./figures/test-" + title + "_" + time_df.iloc[i,0] + ".png")
                plt.close(fig)
                start_pos = i
                continue
            if i == len(time_df.index) - 1:
                # do plot as this is the last data point.
                end_pos = i
                fig = create_plot(time_df, x_df, x_columns, x_colors,
                                  title, limit_vertical_maxval, start_pos, end_pos,
                                  y_df, y_columns, y_colors)
                plt.savefig("./figures/test-" + title + "_" + time_df.iloc[i,0] + ".png")
                plt.close(fig)
    else:
        # format datetime into YYYY/MM/DD HH:MM
        start_pos = 0
        end_pos = 0

        start_datetime_dt = datetime.strptime(start_datetime, "%Y/%m/%d %H:%M")
        end_datetime_dt = datetime.strptime(end_datetime, "%Y/%m/%d %H:%M")
        for index, row in time_df.iterrows():
            temp_datetime_dt = datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S")
            if temp_datetime_dt == end_datetime_dt:
                end_pos = index
            if temp_datetime_dt == start_datetime_dt:
                start_pos = index

        if end_pos > 0:
        # plot data
            create_plot(time_df.iloc[start_pos:end_pos, :], x_df.iloc[start_pos:end_pos, :],
                        x_columns, x_colors,
                        title, limit_vertical_maxval,
                        y_df.iloc[start_pos:end_pos, :], y_columns, y_colors)
            plt.show()
        else:
            print("no data to plot!")


def create_plot(time_df, x_df, x_columns, x_colors,
                title, limit_vertical_maxval,
                y_df=None, y_columns=None, y_colors=None,
                locator=dates.HourLocator(interval=1), plot_mode="plot"):
    """
    generate a matplotlib figure and axis object.

    Args:
        time_matrix: time data
        x_df: dataframe for explanatory variables
        x_columns: columns for explanatory variables
        x_colors: plot colors for explanatory variables
        title: graph title
        limit_vertical_maxval: vertical axis max limitation
        y_df: dataframe for target variables
        y_columns: columns for target variables
        y_colors: plot colors for target variables
        locator: tick locator
        plot_mode: plot or scatter to show the graph. plot as default

    Returns:
        None
    """

    fig, ax1 = plt.subplots(figsize=(len(time_df)/3, 12))
    time_matrix = dates.date2num(
        [datetime.strptime(t, "%Y/%m/%d %H:%M:%S") for t in time_df.values.transpose()[0]])
    for key in x_columns:
        if plot_mode == "plot":
            ax1.plot(time_matrix,
                     x_df.iloc[:, x_df.columns.get_loc(key)],
                     label=x_columns[key], color=x_colors[key])
        elif plot_mode == "scatter":
            ax1.scatter(time_matrix,
                     x_df.iloc[:, x_df.columns.get_loc(key)],
                     label=x_columns[key], color=x_colors[key])
    if y_df is not None:
        for key in y_columns:
            if plot_mode == "plot":
                ax1.plot(time_matrix,
                         y_df.iloc[:, y_df.columns.get_loc(key)],
                         label=y_columns[key], color=y_colors[key])
            elif plot_mode == "scatter":
                ax1.scatter(time_matrix,
                         y_df.iloc[:, y_df.columns.get_loc(key)],
                         label=y_columns[key], color=y_colors[key])
    if limit_vertical_maxval is True: ax1.set_ylim([0, 3.0])
    ax1.legend(loc=2, prop=constants.fp)
    # ax1.set_xticks(time, minor=False)
    ax1.set_xticklabels(time_matrix, rotation=90)
    ax1.tick_params(labelsize=18)
    #ax1.xaxis.set_major_locator(dates.DayLocator(interval=1))
    ax1.xaxis.set_major_locator(locator)
    hfmt = dates.DateFormatter("%Y/%m/%d %H:%M")
    ax1.xaxis.set_major_formatter(hfmt)
    plt.title(title, **constants.sfont)
    return fig, ax1

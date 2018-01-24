#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu

"""
cosntants
"""
from matplotlib import colors as mcolors
from matplotlib.font_manager import FontProperties

# parameters for random forest(hyper parameter)
# high max_features values may lead to high variance.
RAMDON_FOREST_PARAMS = {
    "n_estimators": [100],
    "max_features": [1, "auto", "sqrt", None],
    "max_depth": [1, 5, 10, None],
    "min_samples_leaf": [1, 2, 4, 50]}

MODEL_NAME_LASSO = "Lasso"
MODEL_NAME_ELASTICNET = "ElasticNet"
MODEL_NAME_RIDGE = "Ridge"
MODEL_NAME_RIDGECV = "RidgeCV"
MODEL_NAME_LARS = "Lars"
MODEL_NAME_BAYESIAN = "BayesianRidge"
MODEL_NAME_SGD = "SGD"
MODEL_NAME_RANDOM_FOREST = "RandomForest"

# penalty coefficiency(always positive)
lamdaArray = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10]

fp = FontProperties(fname='ipaexg.ttf', size=20)

sfont = {'fontproperties': fp, 'fontsize': 20, 'color': 'black'}

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
![License](https://img.shields.io/badge/license-Apache-blue.svg)

**timeseriesprocessing is a Python3 library for my daily data science tasks. This library aims to process time series data.**
**timeseriesprocessing is not an ML algorithm library but a tool to do ML easier.**

## What does this library provide
This library is based on scikit-learn >=0.19.
As classification for time series data set is rare and I use deep learning libraries and opencv to process classification problems, classification is not included in the scope of this library.
Currently timeseriesprocessing supports the following modules.
- Time series regression
- Time series clustering
- An ensemble method for stacking regressors

### Time series regression
This is a simple framework to process multi-variables timeseries dataset using regression.<br>

Most time series analysis methods focus on single variable data. It's simple to understand and work with such data. But sometimes our time series dataset may containe multi-varibles. For example, in marketing analysis, profit of a day may not only be decided by the number of customers, but also depend on campaign, CM and so on. It is harder to model such problems and often many of the classical methods do not perform well.<br>

Since regression methods are good at processing multivarible, we can simply turn our timeseries dataset into training dataset for regression by exluding time columns. By doing that, we need to make sure if data at each time point are independent. In another word, our dataset should not be affected by past data.<br>

#### Restrictions
1.In general when using regression methods to make a prediction at a certain time point, data for independent variables at that time point must be avaliable to make predicitons.<br>
2.Data at each time point must be independent.<br>
3.Each independent variable must be unrelated. In many cases multi-variables regression analysis fails to clarify the correlations between the variables.<br>
* For detail about 2. and 3., refer to "Assumption of Independence".<br>
[Here is a good explaination](http://www.statisticshowto.com/assumption-of-independence/)

#### How to use
See [example.html](https://nbviewer.jupyter.org/github/george-j-zhu/timeseriesprocessing/blob/master/sample/regression.ipynb)<br>
In the example, we used only weather data as our features to prediction Energy data from tepco.

### Time series clustering
timeseriesprocessing supports all clustering methods in scikit-learn.<br>
For k-means, timeseriesprocessing can be used to plot an elbow function to determine k. x-means will be added in timeseriesprocessing for selecting the best k for k-means automatically. For other clustering algorithms to determine n_clisters, silhouette analysis can be done easily in timeseriesprocessing. Note that as t-SNE is used to reduce the dimensionality so that clustered data can be plotted onto a 2D plane, if data is very high dimensional, it would be hard to plot clusters correctly.

### An ensemble method for stacking regressors
This module is currently partially copied from mlxtend library and I extended this module to be a able to do normalization and/or scaling for each regressor.

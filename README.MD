# Precessing timeseries problems using Regression
This is a simple framework to process multi-variable timeseries dataset using regression.<br>

Most time series analysis methods focus on single variable data. It's simple to understand<br>
and work with such data. But sometimes our time series dataset may containe multi-varibles.<br>
For example, in marketing analysis, profit of a day may not only be decided by the number<br>
of customers, but also depend on campaign, CM and so on.<br>
It is harder to model such problems and often many of the classical methods do not perform<br>
well.<br>

Since regression methods are good at processing multivarible, we can simplely turn our timeseries<br>
dataset into training dataset for regression by exluding time columns.<br>

## Restrictions
In general when using regression methods, timeseries data for your independent variables must be<br>
avaliable to make predicitons.<br>

## How to use
See example.ipynb

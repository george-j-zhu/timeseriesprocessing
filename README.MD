# Processing timeseries problems using Regression
This is a simple framework to process multi-variables timeseries dataset using regression.<br>

Most time series analysis methods focus on single variable data. It's simple to understand<br>
and work with such data. But sometimes our time series dataset may containe multi-varibles.<br>
For example, in marketing analysis, profit of a day may not only be decided by the number<br>
of customers, but also depend on campaign, CM and so on.<br>
It is harder to model such problems and often many of the classical methods do not perform<br>
well.<br>

Since regression methods are good at processing multivarible, we can simply turn our timeseries<br>
dataset into training dataset for regression by exluding time columns. By doing that, we need<br>
to make sure if data at each time point are independent. In another word, our dataset are not<br>
affected by past data.<br>

## Restrictions
1.In general when using regression methods to make a prediction at a certain time point, data for<br>
independent variables at that time point must be avaliable.<br>
be avaliable to make predicitons.<br>
2.Data at each time point must be independent.<br>
3.Each independent variable must be unrelated. In many cases multi-variables regression analysis<br>
fails to clarify the correlations between the variables.<br>

## How to use
See example.ipynb

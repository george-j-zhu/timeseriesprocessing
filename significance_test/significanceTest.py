#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu

"""
Statistical Hypothesis Test(significance test)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import chi2_contingency
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import kendalltau
import pandas as pd

class SignificanceTest:
    """
    If Data Is Gaussian:
	Use Parametric Statistical Methods
    Else:
	Use Nonparametric Statistical Methods

    """

    def __init__(self):
        """
        constructor
        """
        np.random.seed(0)


    def normality_test(self, data, plot_hist=False):
        """
        Tests whether a data sample has a Normal(Gaussian) distribution.

        Assumptions
            Observations in each sample are independent and identically distributed (iid).

        Interpretation
            H0: the sample has a Gaussian distribution.(null hypothesis)
            H1: the sample does not have a Gaussian distribution.

        Note: Null hypothesis usually means there's no relationship between two data sets.

        Return:
            percentage of success: show the percentage of success. (soft_fail)
                if all failed, we call that hard fail.
                if some failed, we call that soft fail.
        """

        if plot_hist is True:
            plt.hist(data)
            plt.show()

        test_succeeded = 0
        num_tests = 0

        # Shapiro-Wilk Test
        # use p-value only as p-value is easier to evaluate.
        stat, p = shapiro(data)
        num_tests += 1
        # interpret
        alpha = 0.05
        if p > alpha:
            # failed to reject null hypothesis
            test_succeeded += 1

        # D’Agostino’s K^2 test
        stat, p = normaltest(data)
        num_tests += 1
        # interpret
        alpha = 0.05
        if p > alpha:
            test_succeeded += 1

        # Anderson-Darling Test
        result = anderson(data)
        num_tests += 1
        p = 0

        anderson_test_succeeded = True
        for i in range(len(result.critical_values)):
            if result.statistic >= result.critical_values[i]:
                anderson_test_succeeded = False
        if anderson_test_succeeded is True:
            test_succeeded += 1

        test_succeeded_pert = test_succeeded/num_tests

        if test_succeeded_pert == 0:
            print("Samples do not follows a normal distribution. (hard fail)")
        elif test_succeeded_pert == 1:
            print("Samples do follows a normal distribution.")
        else:
            print("Samples may not follows a normal distribution. (soft fail)")

        return test_succeeded_pert


    def correlation_test(self, data1, data2, normal_dist=True, corr_algo="spearman"):
        """
        Checking if two samples are related. The following 3 rank correlation are provided.

        1. Pearson’s Correlation Coefficient
            Tests whether two samples have a monotonic relationship.

            Assumptions
                Observations in each sample are independent and identically distributed (iid).
                Observations in each sample are normally distributed.
                Observations in each sample have the same variance.

            Interpretation
                H0: the two samples are independent.
                H1: there is a dependency between the samples.

        2. Spearman’s Rank Correlation
            Tests whether two samples have a monotonic relationship.

            Assumptions
                Observations in each sample are independent and identically distributed (iid).
                Observations in each sample can be ranked.

            Interpretation
                H0: the two samples are independent.
                H1: there is a dependency between the samples.

        3. Kendall’s Rank Correlation
            Tests whether two samples have a monotonic relationship.

            Assumptions
                Observations in each sample are independent and identically distributed (iid).
                Observations in each sample can be ranked.

            Interpretation
                H0: the two samples are independent.
                H1: there is a dependency between the samples.

        Args:
            data1: input data1
            data2: input data2
            normal_dist: if samples have Normal Distribution.
            corr_algo: rank correlation algorithm name.

        Returns:
            correlations
        """
        algo_name_spearman = "spearman"
        algo_name_kendall = "kendall"

        if normal_dist is True:
            corr, p = pearsonr(data1, data2)
        else:
            if corr_algo == algo_name_spearman:
                corr, p = spearmanr(data1, data2)
            elif corr_algo == algo_name_kendall:
                corr, p = kendalltau(data1, data2)
            else:
                raise ValueError("not supported rank correlation!")

        # interpret the significance
        alpha = 0.05
        if p > alpha:
            print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
        else:
            print('Samples are correlated (reject H0) p=%.3f' % p)

        return corr


    def chi_square_test(self, dataframe):
        """
        Chi-Squared Test
            Tests whether two categorical variables are related or independent.

            Assumptions
                Observations used in the calculation of the contingency table are independent.
                25 or more examples in each cell of the contingency table.

            Interpretation
                H0: the two samples are independent.
                H1: there is a dependency between the samples.
        """

        #              Survived  Pclass     Sex   Age     Fare Cabin Embarked
        # PassengerId
        # 1                   0       3    male  22.0   7.2500   NaN        S
        # 2                   1       1  female  38.0  71.2833   C85        C
        # 3                   1       3  female  26.0   7.9250   NaN        S
        # 4                   1       1  female  35.0  53.1000  C123        S
        # 5                   0       3    male  35.0   8.0500   NaN        S
        #                                \/
        # Pclass    1    2    3
        # Sex
        # female   94   76  144
        # male    122  108  347
        crossed = pd.crosstab(dataframe.A, dataframe.B)
        stat, p, dof, expected = chi2_contingency(crossed)
        alpha = 0.05
        if p > alpha:
            print('Samples are unrelated (fail to reject H0) p=%.3f' % p)
        else:
            print('Samples are related (reject H0) p=%.3f' % p)


    def compare_data_samples(self, data1, data2, normal_dist=True, test_name="t-test"):
        """
        Compare tow data samples.

        Parametric Statistical Hypothesis Tests
        1. Student’s t-test
            Tests whether the means of two independent samples are significantly different.

            Assumptions
                Observations in each sample are independent and identically distributed (iid).
                Observations in each sample are normally distributed.
                Observations in each sample have the same variance.

            Interpretation
                H0: the means of the samples are equal.
                H1: the means of the samples are unequal.

        2. Paired Student’s t-test
            Tests whether the means of two paired samples are significantly different.

            Assumptions
                Observations in each sample are independent and identically distributed (iid).
                Observations in each sample are normally distributed.
                Observations in each sample have the same variance.
                Observations across each sample are paired.

            Interpretation
                H0: the means of the samples are equal.
                H1: the means of the samples are unequal.

        3. Analysis of Variance Test (ANOVA)
            Tests whether the means of two or more independent samples are significantly different.

            Assumptions
                Observations in each sample are independent and identically distributed (iid).
                Observations in each sample are normally distributed.
                Observations in each sample have the same variance.

            Interpretation
                H0: the means of the samples are equal.
                H1: one or more of the means of the samples are unequal.

        Nonparametric Statistical Hypothesis Tests
        4. Mann-Whitney U Test
            Tests whether the distributions of two independent samples are equal or not.

            Assumptions
                Observations in each sample are independent and identically distributed (iid).
                Observations in each sample can be ranked.

            Interpretation
                H0: the distributions of both samples are equal.
                H1: the distributions of both samples are not equal.

        5. Wilcoxon Signed-Rank Test
            Tests whether the distributions of two paired samples are equal or not.

            Assumptions
                Observations in each sample are independent and identically distributed (iid).
                Observations in each sample can be ranked.
                Observations across each sample are paired.

            Interpretation
                H0: the distributions of both samples are equal.
                H1: the distributions of both samples are not equal.

        6. Kruskal-Wallis H Test
            Tests whether the distributions of two or more independent samples are equal or not.

            Assumptions
                Observations in each sample are independent and identically distributed (iid).
                Observations in each sample can be ranked.

            Interpretation
                H0: the distributions of all samples are equal.
                H1: the distributions of one or more samples are not equal.

        7. Friedman Test
            Tests whether the distributions of two or more paired samples are equal or not.

            Assumptions
                Observations in each sample are independent and identically distributed (iid).
                Observations in each sample can be ranked.
                Observations across each sample are paired.

            Interpretation
                H0: the distributions of all samples are equal.
                H1: the distributions of one or more samples are not equal.
        """
        from scipy.stats import ttest_ind
        stat, p = ttest_ind(data1, data2)

        from scipy.stats import ttest_rel
        stat, p = ttest_rel(data1, data2)

        from scipy.stats import f_oneway
        stat, p = f_oneway(data1, data2, ...)

        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(data1, data2)

        from scipy.stats import wilcoxon
        stat, p = wilcoxon(data1, data2)

        from scipy.stats import kruskal
        stat, p = kruskal(data1, data2, ...)

        from scipy.stats import friedmanchisquare
        stat, p = friedmanchisquare(data1, data2, ...)

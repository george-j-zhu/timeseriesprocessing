import unittest

from timeseriesprocessing.significance_test.significanceTest import SignificanceTest

class TestStatisticalHypothesisTest(unittest.TestCase):
    """
    test class of StatisticalHypothesisTest

    test list:
        normality_test
    """

    def test_normality_test(self):
        """
        test select_high_variance_features
        """

        data = load_data()

        expected = 3/3

        tester = SignificanceTest()
        actual = tester.normality_test(data)
        self.assertEqual(expected, actual)

def load_data():
    # generate gaussian data
    from numpy.random import seed
    from numpy.random import randn
    from numpy import mean
    from numpy import std
    # seed the random number generator
    seed(1)
    # generate univariate observations
    data = 5 * randn(100) + 50
    # summarize
    print('mean=%.3f stdv=%.3f' % (mean(data), std(data)))
    return data

if __name__ == "__main__":

    unittest.main()

import numpy as np


class Combiner(object):

    def __init__(self, rule='majority_vote'):
        self.combination_rule = rule

        if rule == 'majority_vote':
            self.rule = majority_vote_rule

        elif rule == 'max':
            self.rule = max_rule

        elif rule == 'min':
            self.rule = min_rule

        elif rule == 'mean':
            self.rule = mean_rule

        elif rule == 'median':
            self.rule = median_rule

        else:
            raise Exception('invalid argument rule for Combiner class')

    def combine(self, results):

        n_samples = results.shape[0]

        out = np.zeros((n_samples,))

        for i in range(n_samples):
            out[i] = self.rule(results[i, :, :])

        return out

def max_rule(probs):
    """ Implements the max rule as defined by [1].
    This rule only makes sense if the classifiers output
    the posterior probabilities for each class.
    Parameters
    ----------
    probs:  Numpy 2d-array with rows representing each class, columns
            representing each classifier and elements representing
            posterior probabilities. Each column should sum up to
            one as a sanity check that the probabilities are valid.
    """

    return probs.max(axis=1).argmax()


def min_rule(probs):
    """ Implements the min rule as defined by [1].
    This rule only makes sense if the classifiers output
    the posterior probabilities for each class.
    Parameters
    ----------
    probs:  Numpy 2d-array with rows representing each class, columns
            representing each classifier and elements representing
            posterior probabilities. Each column should sum up to
            one as a sanity check that the probabilities are valid.
    """

    return probs.min(axis=1).argmax()


def mean_rule(probs):
    """
    Implements the first case of the median rule as defined by [1].
    This rule only makes sense if the classifiers output
    the posterior probabilities for each class.
    Parameters
    ----------
    probs:  Numpy 2d-array with rows representing each class, columns
            representing each classifier and elements representing
            posterior probabilities. Each column should sum up to
            one as a sanity check that the probabilities are valid.
    """

    return probs.mean(axis=1).argmax()


def median_rule(probs):
    """
    Implements the second case of the median rule as defined by [1].
    This rule only makes sense if the classifiers output
    the posterior probabilities for each class.
    Parameters
    ----------
    probs:  Numpy 2d-array with rows representing each class, columns
            representing each classifier and elements representing
            posterior probabilities. Each column should sum up to
            one as a sanity check that the probabilities are valid.
    """

    # numpy array has no median method
    return np.median(probs, axis=1).argmax()


def majority_vote_rule(votes):
    """
    Implements the majority vote rule as defined by [1].
    This rule can always be used, because even if the classifiers output
    posterior probabilities, you can for example, decide to vote for
    the class with the greatest probability. The important thing is to
    transform the classifiers probabilitities/decisions into a matrix
    of votes.
    Parameters
    ----------
    votes:  Numpy 2d-array with rows representing each class, columns
            representing each classifier and elements representing
            votes (binary). Each column should sum up to one (i.e.
            a classifier can only vote for one class).
    """

    return votes.sum(axis=1).argmax()

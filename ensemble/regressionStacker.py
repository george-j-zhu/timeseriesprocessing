import numpy as np
from .regressionLayer import EnsembleLayer

from sklearn.model_selection import StratifiedKFold


class EnsembleStack(object):

    def __init__(self, cv=5):
        self.layers = []
        self.cv = cv

    def add_layer(self, ensemble_layer):
        """
        add a layer to this stack
        """
        if isinstance(ensemble_layer, EnsembleLayer):
            self.layers.append(ensemble_layer)
        else:
            raise Exception('not an Ensemble object')

    def fit_layer(self, layer_idx, X, y):
        """
        fit a layer by layer_idx
        """
        if layer_idx >= len(self.layers):
            return
        elif layer_idx == len(self.layers) - 1:
            self.layers[layer_idx].fit(X, y)
        else:

            n_models = self.layers[layer_idx].size()
            output = np.zeros((X.shape[0], n_models))
            skf = StratifiedKFold(self.cv)
            for train_idx, test_idx in skf.split(X, y):
                self.layers[layer_idx].fit(X[train_idx], y[train_idx])
                out = self.layers[layer_idx].output(X[test_idx])
                output[test_idx, :] = out

            self.layers[layer_idx].fit(X, y)
            self.fit_layer(layer_idx + 1, output, y)

    def fit(self, X, y):
        """
        fit this stack
        """
        if self.cv > 1:
            self.fit_layer(0, X, y)
        else:
            X_ = X
            for layer in self.layers:
                layer.fit(X_, y)
                out = layer.output(X_)
                X_ = out[:, 1:, :].reshape(
                    out.shape[0], (out.shape[1] - 1) * out.shape[2])

        return self

    def output(self, X):

        out = None
        for layer in self.layers:
            if out is None:
                out = layer.output(X)
            else:
                out = layer.output(out)
        return out


class EnsembleStackRegressor(object):

    def __init__(self, stack):
        self.stack = stack

    def fit(self, X, y):
        """
        fit this regressor
        """
        self.stack.fit(X, y)
        return self

    def predict(self, X):
        """
        make predictions
        """
        return self.stack.output(X)

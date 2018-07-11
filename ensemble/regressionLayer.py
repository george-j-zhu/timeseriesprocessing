import numpy as np


class EnsembleLayer(object):

    def __init__(self, models=None, normalizers=None):

        if models is None:
            self.models = []
        else:
            self.models = models

        if normalizers is None:
            self.normalizers = []
        else:
            self.normalizers = normalizers

    def add(self, model, normalizer):
        """
        add new model
        """
        self.models.append(model)
        self.normalizers.append(normalizer)

    def add_models(self, models, normalizers):
        """
        add new models
        """
        self.models = self.models + models
        self.normalizers = self.normalizers + normalizers

    def add_ensemble(self, ensemble):
        """
        add an ensemble
        """
        self.models = self.add_models(ensemble.models, ensemble.normalizers)

    def output(self, X):

        out = np.zeros((X.shape[0], len(self.models)))
        for i, clf in enumerate(self.models):
            out[:, i] = clf.predict(X)
        return out

    def size(self):
        return len(self.models)

    def fit(self, X, y):

        for clf in self.models:
            clf.fit(X, y)

        return self

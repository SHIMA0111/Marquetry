import unittest

import numpy as np

from marquetry.ml import ClassificationTree, RandomForest, RegressionTree


def linearly_separable_classification(samples=120, seed=0):
    np.random.seed(seed)
    x = np.random.randn(samples, 2)
    t = (x[:, 0] + x[:, 1] > 0).astype(int)

    return x, t


class TestClassificationTree(unittest.TestCase):

    def test_train_accuracy_on_separable_data(self):
        x, t = linearly_separable_classification()

        tree = ClassificationTree(max_depth=6, criterion="gini", seed=1)
        tree.fit(x, t)
        predict = tree.predict(x)

        accuracy = float((predict.data == t).mean())
        self.assertGreater(accuracy, 0.9)

    def test_entropy_criterion(self):
        x, t = linearly_separable_classification(seed=2)

        tree = ClassificationTree(max_depth=6, criterion="entropy", seed=1)
        tree.fit(x, t)
        predict = tree.predict(x)

        accuracy = float((predict.data == t).mean())
        self.assertGreater(accuracy, 0.9)

    def test_invalid_criterion_rejected(self):
        with self.assertRaises(Exception):
            tree = ClassificationTree(criterion="rss")
            tree.fit(*linearly_separable_classification())


class TestRegressionTree(unittest.TestCase):

    def test_fits_step_function(self):
        np.random.seed(0)
        x = np.random.randn(100, 1)
        t = np.where(x[:, 0] > 0.0, 3.0, -3.0)

        tree = RegressionTree(max_depth=4, criterion="rss", seed=1)
        tree.fit(x, t)
        predict = tree.predict(x)

        mse = float(((predict.data - t) ** 2).mean())
        self.assertLess(mse, 0.5)


class TestRandomForest(unittest.TestCase):

    def test_classification_accuracy(self):
        x, t = linearly_separable_classification(samples=150, seed=3)

        model = RandomForest(n_trees=9, max_depth=6, seed=5)
        model.fit(x, t)
        predict = model.predict(x)

        accuracy = float((predict.data == t).mean())
        self.assertGreater(accuracy, 0.7)

    def test_regression_mode(self):
        np.random.seed(1)
        x = np.random.randn(120, 2)
        t = x[:, 0] * 2.0 + x[:, 1]

        model = RandomForest(n_trees=5, target_type="regression", criterion="rss", max_depth=6, seed=2)
        model.fit(x, t)
        predict = model.predict(x)

        mse = float(((predict.data - t) ** 2).mean())
        self.assertLess(mse, float(t.var()))

    def test_invalid_target_type_rejected(self):
        with self.assertRaises(ValueError):
            RandomForest(target_type="cluster")

    def test_predict_before_fit_rejected(self):
        model = RandomForest(n_trees=2)

        with self.assertRaises(Exception):
            model.predict(np.random.randn(4, 2))

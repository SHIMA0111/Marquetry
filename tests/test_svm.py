import os
import tempfile
import unittest

import numpy as np

from marquetry.ml import SVM


def separable_data(samples=100, seed=1):
    """Linearly separable 2-class data with a clear margin, labels in {-1, 1}."""
    rng = np.random.RandomState(seed)
    x = rng.randn(samples, 2)
    true_w = np.array([1.5, -1.0])

    margin = x @ true_w
    x = x[np.abs(margin) > 0.5]
    t = np.where(x @ true_w > 0, 1, -1)

    return x, t


class TestSVM(unittest.TestCase):

    def test_soft_margin_accuracy(self):
        x, t = separable_data()

        model = SVM(c=1.0)
        model.fit(x, t)
        predict = model.predict(x)

        accuracy = float((predict.data == t).mean())
        self.assertGreater(accuracy, 0.95)

    def test_hard_margin_accuracy(self):
        x, t = separable_data(seed=2)

        model = SVM(c=None)
        model.fit(x, t)
        predict = model.predict(x)

        accuracy = float((predict.data == t).mean())
        self.assertGreater(accuracy, 0.95)

    def test_zero_one_labels_round_trip(self):
        x, t = separable_data(seed=3)
        t01 = (t > 0).astype(int)

        model = SVM()
        model.fit(x, t01)
        predict = model.predict(x)

        self.assertTrue(set(np.unique(predict.data)).issubset({0, 1}))
        accuracy = float((predict.data == t01).mean())
        self.assertGreater(accuracy, 0.95)

    def test_decision_function_sign_matches_predict(self):
        x, t = separable_data(seed=4)

        model = SVM()
        model.fit(x, t)

        decision = model.decision_function(x)
        predict = model.predict(x)

        np.testing.assert_array_equal(np.where(decision > 0, 1, -1), predict.data)

    def test_save_load_roundtrip(self):
        x, t = separable_data(seed=5)

        model = SVM()
        model.fit(x, t)
        predict_before = model.predict(x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "svm.npz")
            model.save_params(path)

            restored = SVM()
            restored.load_params(path)

        predict_after = restored.predict(x)
        np.testing.assert_array_equal(predict_before.data, predict_after.data)

    def test_multiclass_rejected(self):
        x = np.random.randn(30, 2)
        t = np.random.randint(0, 3, 30)

        model = SVM()
        with self.assertRaises(ValueError):
            model.fit(x, t)

    def test_predict_before_fit_rejected(self):
        model = SVM()

        with self.assertRaises(RuntimeError):
            model.predict(np.random.randn(4, 2))

    def test_invalid_c_rejected(self):
        with self.assertRaises(ValueError):
            SVM(c=-1.0)

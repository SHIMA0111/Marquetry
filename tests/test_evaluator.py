import unittest

import numpy as np
from sklearn import metrics

from marquetry import functions


class TestAccuracy(unittest.TestCase):

    def test_forward1(self):
        y = np.array([[0.8, 0.1, 0.05, 0.05], [0.1, 0.01, 0.59, 0.3], [0.0, 0.01, 0.32, 0.67]])
        t = np.array([0, 2, 2])

        expected = np.array(2 / 3)
        accuracy_score = functions.accuracy(y, t)

        self.assertEqual(expected, accuracy_score.data)

    def test_forward2(self):
        y = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.01, 0.59, 0.3],
            [0.0, 0.01, 0.32, 0.67],
            [0.9, 0.03, 0.03, 0.04]])
        t = np.array([0, 3, 3, 1])

        ignore_class = 1

        expected = np.array(2 / 3)
        accuracy_score = functions.accuracy(y, t, ignore_class)

        self.assertEqual(expected, accuracy_score.data)

    def test_forward3(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([1, 0, 0, 1, 1, 1])

        expected = np.array(4 / 6)
        accuracy_score = functions.binary_accuracy(y, t)

        self.assertEqual(expected, accuracy_score.data)

    def test_forward4(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([1, 0, 0, 1, 1, 1])

        expected = np.array(1.)
        accuracy_score = functions.binary_accuracy(y, t, threshold=0.6)

        self.assertEqual(expected, accuracy_score.data)


class TestFScore(unittest.TestCase):

    def test_forward1(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([1, 0, 0, 1, 1, 1])

        y_pred = np.asarray(y >= .7, dtype="i")

        expected = metrics.f1_score(t, y_pred)
        f_score = functions.f_score(y, t)

        self.assertEqual(expected, f_score.data)

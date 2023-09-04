import unittest

import numpy as np
from sklearn import metrics

from marquetry import functions
from marquetry.utils import array_close


class TestAccuracy(unittest.TestCase):

    def test_forward1(self):
        y = np.array([[0.8, 0.1, 0.05, 0.05], [0.1, 0.01, 0.59, 0.3], [0.0, 0.01, 0.32, 0.67]])
        t = np.array([0, 2, 2])

        expected = np.array(2 / 3)
        accuracy_score = functions.evaluation.accuracy(y, t)

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
        accuracy_score = functions.evaluation.accuracy(y, t, ignore_class)

        self.assertEqual(expected, accuracy_score.data)

    def test_forward3(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([1, 0, 0, 1, 1, 1])

        expected = np.array(4 / 6)
        accuracy_score = functions.evaluation.binary_accuracy(y, t)

        self.assertEqual(expected, accuracy_score.data)

    def test_forward4(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([1, 0, 0, 1, 1, 1])

        expected = np.array(1.)
        accuracy_score = functions.evaluation.binary_accuracy(y, t, threshold=0.6)

        self.assertEqual(expected, accuracy_score.data)


class TestFScore(unittest.TestCase):

    def test_forward1(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([1, 0, 0, 1, 1, 1])

        y_pred = np.asarray(y >= .7, dtype="i")

        expected = metrics.f1_score(t, y_pred)
        f_score = functions.evaluation.f_score(y, t)

        self.assertEqual(expected, f_score.data)

    def test_forward2(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([0, 0, 0, 0, 0, 0])

        y_pred = np.asarray(y >= .7, dtype="i")

        expected = metrics.f1_score(t, y_pred)
        f_score = functions.evaluation.f_score(y, t)

        self.assertEqual(expected, f_score.data)

    def test_forward3(self):
        y = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.01, 0.59, 0.3],
            [0.0, 0.01, 0.32, 0.67],
            [0.9, 0.03, 0.03, 0.04]])
        t = np.array([0, 3, 3, 1])

        target_class = 3
        y_pred = np.argmax(y, axis=1)

        pred_trans = np.asarray(y_pred == target_class, dtype="i")
        correct_trans = np.asarray(t == target_class, dtype="i")

        expected = metrics.f1_score(correct_trans, pred_trans)
        multi_f_score = functions.evaluation.multi_f_score(y, t, target_class=target_class)

        self.assertEqual(expected, multi_f_score.data)


class TestClassificationError(unittest.TestCase):

    def test_forward1(self):
        y = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.01, 0.59, 0.3],
            [0.0, 0.01, 0.32, 0.67],
            [0.9, 0.03, 0.03, 0.04]])
        t = np.array([0, 3, 3, 1])

        expected = np.array(2 / 4)
        cl_error = functions.evaluation.classification_error(y, t)

        self.assertEqual(expected, cl_error.data)

    def test_forward2(self):
        y = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.01, 0.59, 0.3],
            [0.0, 0.01, 0.32, 0.67],
            [0.9, 0.03, 0.03, 0.04]])
        t = np.array([0, 3, 3, 1])

        ignore_label = 1

        expected = np.array(1 / 3)
        cl_error = functions.evaluation.classification_error(y, t, ignore_label)

        self.assertEqual(expected, cl_error.data)

    def test_forward3(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([1, 0, 0, 1, 1, 1])

        expected = np.array(1 / 3)
        cl_error = functions.evaluation.binary_classification_error(y, t)

        self.assertEqual(expected, cl_error.data)

    def test_forward4(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([1, 0, 0, 1, 1, 1])

        expected = np.array(0 / 6)
        cl_error = functions.evaluation.binary_classification_error(y, t, threshold=0.6)

        self.assertEqual(expected, cl_error.data)


class TestPrecision(unittest.TestCase):

    def test_forward1(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([1, 0, 0, 1, 1, 1])

        y_pred = np.asarray(y >= .7, dtype="i")

        expected = metrics.precision_score(t, y_pred)
        precision = functions.evaluation.precision(y, t)

        self.assertEqual(expected, precision.data)

    def test_forward2(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([0, 0, 0, 0, 0, 0])

        y_pred = np.asarray(y >= .7, dtype="i")

        expected = metrics.precision_score(t, y_pred)
        precision = functions.evaluation.precision(y, t)

        self.assertEqual(expected, precision.data)

    def test_forward3(self):
        y = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.01, 0.59, 0.3],
            [0.0, 0.01, 0.32, 0.67],
            [0.9, 0.03, 0.03, 0.04]])
        t = np.array([0, 3, 3, 1])

        target_class = 3
        y_pred = np.argmax(y, axis=1)

        pred_trans = np.asarray(y_pred == target_class, dtype="i")
        correct_trans = np.asarray(t == target_class, dtype="i")

        expected = metrics.precision_score(correct_trans, pred_trans)
        multi_precision = functions.evaluation.multi_precision(y, t, target_class=target_class)

        self.assertEqual(expected, multi_precision.data)


class TestRecall(unittest.TestCase):

    def test_forward1(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([1, 0, 0, 1, 1, 1])

        y_pred = np.asarray(y >= .7, dtype="i")

        expected = metrics.recall_score(t, y_pred)
        recall = functions.evaluation.recall(y, t)

        self.assertEqual(expected, recall.data)

    def test_forward2(self):
        y = np.array([0.89, 0.03, 0.56, 0.61, 0.65, 0.98])
        t = np.array([0, 0, 0, 0, 0, 0])

        expected = np.array(0.)
        recall = functions.evaluation.recall(y, t)

        self.assertEqual(expected, recall.data)

    def test_forward3(self):
        y = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.01, 0.59, 0.3],
            [0.0, 0.01, 0.32, 0.67],
            [0.9, 0.03, 0.03, 0.04]])
        t = np.array([0, 3, 3, 1])

        target_class = 3
        y_pred = np.argmax(y, axis=1)

        pred_trans = np.asarray(y_pred == target_class, dtype="i")
        correct_trans = np.asarray(t == target_class, dtype="i")

        expected = metrics.recall_score(correct_trans, pred_trans)
        multi_recall = functions.evaluation.multi_recall(y, t, target_class=target_class)

        self.assertEqual(expected, multi_recall.data)


class TestMAE(unittest.TestCase):

    def test_forward1(self):
        y = np.array([1001.2, 211.3, 4500.1, 200.0])
        t = np.array([1231., 199.2, 4328., 176.])

        sk_mae = metrics.mean_absolute_error(t, y)
        mae = functions.evaluation.mean_absolute_error(y, t)

        self.assertEqual(sk_mae, mae.data)

    def test_forward2(self):
        y = np.array([[1001.2, 211.3, 4500.1, 200.0], [1421.2, 231.3, 40.1, 140.0]])
        t = np.array([[1231., 199.2, 4328., 176.], [1312., 220.2, 43.4, 116.]])

        sk_mae = metrics.mean_absolute_error(t, y, multioutput="raw_values")
        mae = functions.evaluation.mean_absolute_error(y, t, multi_output="raw_values")

        self.assertTrue(array_close(sk_mae, mae.data))

    def test_forward3(self):
        y = np.array([[1001.2, 211.3, 4500.1, 200.0], [1421.2, 231.3, 40.1, 140.0]])
        t = np.array([[1231., 199.2, 4328., 176.], [1312., 220.2, 43.4, 116.]])

        sk_mae = metrics.mean_absolute_error(t, y)
        mae = functions.evaluation.mean_absolute_error(y, t)

        self.assertTrue(array_close(sk_mae, mae.data))


class TestMSE(unittest.TestCase):

    def test_forward1(self):
        y = np.array([1001.2, 211.3, 4500.1, 200.0])
        t = np.array([1231., 199.2, 4328., 176.])

        sk_mse = metrics.mean_squared_error(t, y)
        mse = functions.evaluation.mean_squared_error(y, t)

        self.assertEqual(sk_mse, mse.data)

    def test_forward2(self):
        y = np.array([[1001.2, 211.3, 4500.1, 200.0], [1421.2, 231.3, 40.1, 140.0]])
        t = np.array([[1231., 199.2, 4328., 176.], [1312., 220.2, 43.4, 116.]])

        sk_mse = metrics.mean_squared_error(t, y, multioutput="raw_values")
        mse = functions.evaluation.mean_squared_error(y, t, multi_output="raw_values")

        self.assertTrue(array_close(sk_mse, mse.data))

    def test_forward3(self):
        y = np.array([[1001.2, 211.3, 4500.1, 200.0], [1421.2, 231.3, 40.1, 140.0]])
        t = np.array([[1231., 199.2, 4328., 176.], [1312., 220.2, 43.4, 116.]])

        sk_mse = metrics.mean_squared_error(t, y)
        mse = functions.evaluation.mean_squared_error(y, t)

        self.assertTrue(array_close(sk_mse, mse.data))


class TestR2Score(unittest.TestCase):

    def test_forward1(self):
        y = np.array([1001.2, 211.3, 4500.1, 200.0])
        t = np.array([1231., 199.2, 4328., 176.])

        sk_r2 = metrics.r2_score(t, y)
        r2 = functions.evaluation.r2_score(y, t)

        self.assertEqual(sk_r2, r2.data)

    def test_forward2(self):
        y = np.array([[1001.2, 211.3, 4500.1, 200.0], [1421.2, 231.3, 40.1, 140.0]])
        t = np.array([[1231., 199.2, 4328., 176.], [1312., 220.2, 43.4, 116.]])

        sk_r2 = metrics.r2_score(t, y, multioutput="raw_values")
        r2 = functions.evaluation.r2_score(y, t, multi_output="raw_values")

        self.assertTrue(array_close(sk_r2, r2.data))

    def test_forward3(self):
        y = np.array([[1001.2, 211.3, 4500.1, 200.0], [1421.2, 231.3, 40.1, 140.0]])
        t = np.array([[1231., 199.2, 4328., 176.], [1312., 220.2, 43.4, 116.]])

        sk_r2 = metrics.r2_score(t, y)
        r2 = functions.evaluation.r2_score(y, t)

        self.assertTrue(array_close(sk_r2, r2.data))


class TestRMSE(unittest.TestCase):

    def test_forward1(self):
        y = np.array([1001.2, 211.3, 4500.1, 200.0])
        t = np.array([1231., 199.2, 4328., 176.])

        sk_rmse = metrics.mean_squared_error(t, y, squared=False)
        rmse = functions.evaluation.root_mean_squared_error(y, t)

        self.assertEqual(sk_rmse, rmse.data)

    def test_forward2(self):
        y = np.array([[1001.2, 211.3, 4500.1, 200.0], [1421.2, 231.3, 40.1, 140.0]])
        t = np.array([[1231., 199.2, 4328., 176.], [1312., 220.2, 43.4, 116.]])

        sk_rmse = metrics.mean_squared_error(t, y, multioutput="raw_values", squared=False)
        rmse = functions.evaluation.root_mean_squared_error(y, t, multi_output="raw_values")

        self.assertTrue(array_close(sk_rmse, rmse.data))

    def test_forward3(self):
        y = np.array([[1001.2, 211.3, 4500.1, 200.0], [1421.2, 231.3, 40.1, 140.0]])
        t = np.array([[1231., 199.2, 4328., 176.], [1312., 220.2, 43.4, 116.]])

        sk_rmse = metrics.mean_squared_error(t, y, squared=False)
        rmse = functions.evaluation.root_mean_squared_error(y, t)

        self.assertTrue(array_close(sk_rmse, rmse.data))


class TestRAE(unittest.TestCase):

    def test_forward1(self):
        y = np.array([1001.2, 211.3, 4500.1, 200.0])
        t = np.array([1231., 199.2, 4328., 176.])

        expected = np.absolute(y - t).sum() / np.absolute(t - t.mean()).sum()
        rae = functions.evaluation.relative_absolute_error(y, t)

        self.assertEqual(expected, rae.data)

    def test_forward2(self):
        y = np.array([[1001.2, 211.3, 4500.1, 200.0], [1421.2, 231.3, 40.1, 140.0]])
        t = np.array([[1231., 199.2, 4328., 176.], [1312., 220.2, 43.4, 116.]])

        expected = np.absolute(y - t).sum(axis=0) / np.absolute(t - t.mean(axis=0)).sum(axis=0)
        rae = functions.evaluation.relative_absolute_error(y, t, multi_output="raw_values")

        self.assertTrue(array_close(expected, rae.data))

    def test_forward3(self):
        y = np.array([[1001.2, 211.3, 4500.1, 200.0], [1421.2, 231.3, 40.1, 140.0]])
        t = np.array([[1231., 199.2, 4328., 176.], [1312., 220.2, 43.4, 116.]])

        expected = (np.absolute(y - t).sum(axis=0) / np.absolute(t - t.mean(axis=0)).sum(axis=0)).mean()
        rae = functions.evaluation.relative_absolute_error(y, t)

        self.assertTrue(array_close(expected, rae.data))


class TestRSE(unittest.TestCase):

    def test_forward1(self):
        y = np.array([1001.2, 211.3, 4500.1, 200.0])
        t = np.array([1231., 199.2, 4328., 176.])

        expected = np.square(y - t).sum() / np.square(t - t.mean()).sum()
        rse = functions.evaluation.relative_squared_error(y, t)

        self.assertEqual(expected, rse.data)

    def test_forward2(self):
        y = np.array([[1001.2, 211.3, 4500.1, 200.0], [1421.2, 231.3, 40.1, 140.0]])
        t = np.array([[1231., 199.2, 4328., 176.], [1312., 220.2, 43.4, 116.]])

        expected = np.square(y - t).sum(axis=0) / np.square(t - t.mean(axis=0)).sum(axis=0)
        rse = functions.evaluation.relative_squared_error(y, t, multi_output="raw_values")

        self.assertTrue(array_close(expected, rse.data))

    def test_forward3(self):
        y = np.array([[1001.2, 211.3, 4500.1, 200.0], [1421.2, 231.3, 40.1, 140.0]])
        t = np.array([[1231., 199.2, 4328., 176.], [1312., 220.2, 43.4, 116.]])

        expected = (np.square(y - t).sum(axis=0) / np.square(t - t.mean(axis=0)).sum(axis=0)).mean()
        rse = functions.evaluation.relative_squared_error(y, t)

        self.assertTrue(array_close(expected, rse.data))

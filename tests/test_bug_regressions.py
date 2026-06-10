"""Regression tests for the bug-fix batch applied on the engine-refactoring branch.

Each test corresponds to a confirmed bug; see the test names and comments
for which behavior they pin down.
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

import marquetry
import marquetry.functions as funcs
from marquetry import Container
from marquetry.functions.connection.convolution_2d import Convolution2D
from marquetry.functions.connection.convolution_2d_grad_w import Conv2DGradW
from marquetry.utils import array_close, array_equal, gradient_check


class TestComparisonOperators(unittest.TestCase):
    """`__gt__` used to return the result of `<`."""

    def test_gt(self):
        x = Container(np.array([1, 5, 3]))
        y = np.array([2, 4, 3])

        self.assertEqual((x > y).data.tolist(), [False, True, False])

    def test_lt(self):
        x = Container(np.array([1, 5, 3]))
        y = np.array([2, 4, 3])

        self.assertEqual((x < y).data.tolist(), [True, False, False])

    def test_gt_lt_consistency(self):
        x = Container(np.array([1.0, 2.0]))
        y = Container(np.array([2.0, 1.0]))

        self.assertEqual((x > y).data.tolist(), (y < x).data.tolist())


class TestAstype(unittest.TestCase):
    """Non-inplace astype used to return a raw ndarray."""

    def test_returns_container(self):
        x = Container(np.array([1, 2, 3], dtype="int32"))
        y = x.astype("float64")

        self.assertIsInstance(y, Container)
        self.assertEqual(y.dtype, np.dtype("float64"))
        self.assertEqual(x.dtype, np.dtype("int32"))


class TestRandomInt(unittest.TestCase):
    """random_int(high omitted) used to raise ValueError against its own docs."""

    def test_high_omitted(self):
        x = marquetry.random_int(10, size=100)

        self.assertTrue(((x.data >= 1) & (x.data < 10)).all())


class TestMeanSquaredErrorUpstreamGrad(unittest.TestCase):
    """MSE backward used to multiply the upstream gradient twice into grad_x1."""

    def test_scaled_loss(self):
        x0 = Container(np.random.randn(7))
        x1 = Container(np.random.randn(7))

        loss = funcs.mean_squared_error(x0, x1) * 3.0
        loss.backward()

        expected = 3.0 * 2.0 * (x0.data - x1.data) / x0.data.size
        self.assertTrue(array_close(x0.grad.data, expected))
        self.assertTrue(array_close(x1.grad.data, -expected))


class TestDropoutBackward(unittest.TestCase):
    """Dropout backward used to drop the 1/(1-rate) scale, and eval-mode backward raised."""

    def test_train_grad_scale(self):
        np.random.seed(0)
        x = Container(np.ones(1000, dtype=np.float32))

        y = funcs.dropout(x, 0.5)
        y.backward()

        self.assertTrue(array_equal(x.grad.data, y.data))

    def test_eval_grad_identity(self):
        x = Container(np.ones(10, dtype=np.float32))

        with marquetry.test_mode():
            y = funcs.dropout(x, 0.5)
        y.backward()

        self.assertTrue(array_equal(x.grad.data, np.ones(10, dtype=np.float32)))


def get_bn_params(channels, dtype=np.float64):
    gamma = np.random.randn(channels).astype(dtype)
    beta = np.random.randn(channels).astype(dtype)
    mean = np.random.randn(channels).astype(dtype)
    var = np.abs(np.random.randn(channels).astype(dtype))

    return gamma, beta, mean, var


class TestBatchNormBackward(unittest.TestCase):
    """4D backward divided by N instead of N*H*W, and eval-mode backward used stale stats."""

    def test_backward_4d(self):
        x = np.random.randn(3, 2, 4, 5)
        gamma, beta, mean, var = get_bn_params(2)
        f = lambda x_in: funcs.batch_normalization(x_in, gamma, beta, mean, var)

        self.assertTrue(gradient_check(f, x))

    def test_backward_eval_mode(self):
        x = Container(np.random.randn(8, 3))
        gamma, beta, mean, var = get_bn_params(3)

        with marquetry.test_mode():
            y = funcs.batch_normalization(x, gamma, beta, mean, var)
        y.backward()

        expected = np.ones_like(x.data) * gamma / np.sqrt(var + 1e-15)
        self.assertTrue(array_close(x.grad.data, expected))


class TestConv2DGradWDoubleBackward(unittest.TestCase):
    """Conv2DGradW.backward used its own output instead of the upstream gradient."""

    def test_uses_upstream_grad(self):
        x = np.random.randn(2, 3, 5, 5)
        w = np.random.randn(4, 3, 3, 3)
        conv = Convolution2D(stride=1, pad=1)
        y = conv(Container(x), Container(w), None)

        coeff = Container(np.random.randn(*w.shape))
        gy = Container(np.random.randn(*y.shape))
        f = lambda gy_in: funcs.sum(Conv2DGradW(conv)(x, gy_in) * coeff)

        self.assertTrue(gradient_check(f, gy))


class TestMomentumSGD(unittest.TestCase):
    """MomentumSGD never used its learning rate."""

    def test_matches_torch_sgd(self):
        w0 = np.random.randn(4, 3).astype(np.float32)

        layer = marquetry.Layer()
        layer.p = marquetry.Parameter(w0.copy())
        optimizer = marquetry.optimizers.MomentumSGD(lr=0.1, decay=0.9).prepare(layer)

        torch_param = torch.nn.Parameter(torch.tensor(w0))
        torch_optimizer = torch.optim.SGD([torch_param], lr=0.1, momentum=0.9)

        for _ in range(5):
            grad = np.random.randn(4, 3).astype(np.float32)
            layer.p.grad = Container(grad.copy())
            optimizer.update()

            torch_param.grad = torch.tensor(grad)
            torch_optimizer.step()

        self.assertTrue(array_close(layer.p.data, torch_param.detach().numpy()))


class TestGRUFirstStep(unittest.TestCase):
    """The first step skipped the update gate, diverging from a zero initial state."""

    def test_first_step_equals_zero_state(self):
        gru = marquetry.layers.GRU(8)
        x = np.random.randn(4, 6).astype(np.float32)

        y_fresh = gru(x)

        gru.reset_state()
        gru.set_state(Container(np.zeros((4, 8), dtype=np.float32)))
        y_zero_state = gru(x)

        self.assertTrue(array_close(y_fresh.data, y_zero_state.data))


class TestEmbeddingFreeze(unittest.TestCase):
    """set_embedding_vector used to wipe _params, breaking save_params."""

    def test_freeze_keeps_saving(self):
        embedding = marquetry.layers.Embedding(10, 4)
        vector = np.random.randn(10, 4)
        embedding.set_embedding_vector(vector)

        self.assertEqual(list(embedding.params()), [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "embedding.npz")
            embedding.save_params(path)

            restored = marquetry.layers.Embedding(10, 4)
            restored.load_params(path)

        self.assertTrue(array_equal(restored.w.data, vector))


class TestLog2Backward(unittest.TestCase):
    """Log2/Log10 backward built graph nodes for ln(2) and promoted float32 to float64."""

    def test_gradient(self):
        x = Container(np.random.rand(6) + 0.5)

        self.assertTrue(gradient_check(funcs.log2, x))

    def test_grad_keeps_float32(self):
        x = Container(np.random.rand(4).astype(np.float32) + 0.5)

        y = funcs.log2(x)
        y.backward()

        self.assertEqual(x.grad.data.dtype, np.float32)


class TestColumnStandardize(unittest.TestCase):
    """min/max statistics were swapped (scaling was inverted); zero range produced NaN."""

    def test_minmax_direction_and_zero_range(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with marquetry.using_config("CACHE_DIR", tmp_dir):
                data = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 5.0, 5.0, 5.0]})
                standardize = marquetry.preprocesses.ColumnStandardize(
                    ["a", "b"], "test_standardize_direction", True)
                result = standardize(data)

        np.testing.assert_allclose(result["a"].to_numpy(), [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        np.testing.assert_allclose(result["b"].to_numpy(), [0.0, 0.0, 0.0, 0.0])


class TestColumnNormalize(unittest.TestCase):
    """Zero standard deviation produced NaN/inf."""

    def test_zero_std_guard(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with marquetry.using_config("CACHE_DIR", tmp_dir):
                data = pd.DataFrame({"a": [3.0, 3.0, 3.0]})
                normalize = marquetry.preprocesses.ColumnNormalize(["a"], "test_normalize_zero_std", True)
                result = normalize(data)

        self.assertTrue(np.isfinite(result["a"].to_numpy()).all())


class TestMissImputation(unittest.TestCase):
    """Numeric columns were imputed with the category-method statistic."""

    def test_numeric_method_used(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with marquetry.using_config("CACHE_DIR", tmp_dir):
                data = pd.DataFrame({"num": [1.0, 2.0, np.nan, 9.0]})
                imputation = marquetry.preprocesses.MissImputation(
                    [], ["num"], "test_imputation_numeric", True, numeric_method="median")
                result = imputation(data)

        # median of [1, 2, 9] is 2.0; the old code used mode (1.0).
        self.assertEqual(result["num"].iloc[2], 2.0)


class TestLabelEncodeUnknownDetection(unittest.TestCase):
    """Unknown categories were only detected in the last column."""

    def test_unknown_in_first_column(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with marquetry.using_config("CACHE_DIR", tmp_dir):
                train = pd.DataFrame({"c1": ["a", "b", "a"], "c2": ["x", "y", "x"]})
                encoder = marquetry.preprocesses.LabelEncode(["c1", "c2"], "test_label_unknown", True)
                encoder(train)

                test = pd.DataFrame({"c1": ["a", "b", "z"], "c2": ["x", "y", "x"]})
                test_encoder = marquetry.preprocesses.LabelEncode(["c1", "c2"], "test_label_unknown", False)
                result = test_encoder(test)

        self.assertEqual(result["c1"].iloc[2], -1)


class TestRandomForestLabels(unittest.TestCase):
    """Class labels were used as column indexes, breaking non-contiguous labels."""

    def test_non_contiguous_labels(self):
        np.random.seed(0)
        x = np.random.randn(60, 5)
        t = np.where(x[:, 0] > 0.0, 5, 2)

        model = marquetry.ml.RandomForest(n_trees=3, seed=1)
        model.fit(x, t)
        predict = model.predict(x)

        self.assertTrue(set(np.unique(predict.data)).issubset({2, 5}))


class TestCompose(unittest.TestCase):
    """Compose() with the default argument used to raise TypeError."""

    def test_default_args(self):
        compose = marquetry.transformers.Compose()

        self.assertEqual(compose(42), 42)


class TestScalarPromotion(unittest.TestCase):
    """Under NumPy 2, Python-scalar operands promoted float32 Containers to float64."""

    def test_float32_preserved_with_python_scalars(self):
        x = Container(np.ones(3, dtype=np.float32))

        for y in (x * 2.0, x + 1.0, x - 0.5, x / 2.0, 2.0 * x, 1.0 + x, 1.0 - x, 2.0 / x):
            self.assertEqual(y.dtype, np.float32)

    def test_int_with_float_scalar_promotes(self):
        x = Container(np.arange(3))

        self.assertEqual((x * 0.5).dtype, np.float64)


class TestSoftmaxCrossEntropyGradDtype(unittest.TestCase):
    """The one-hot matrix in backward used the integer target dtype, promoting grads to float64."""

    def test_grad_keeps_float32(self):
        x = Container(np.random.randn(4, 3).astype(np.float32))
        t = np.array([0, 2, 1, 1])

        loss = funcs.softmax_cross_entropy(x, t)
        loss.backward()

        self.assertEqual(x.grad.data.dtype, np.float32)


class TestBinaryMultiMetrics(unittest.TestCase):
    """Multi-class metrics rejected two-class logits via `assert unique > 2`."""

    def test_two_class_logits_accepted(self):
        y = np.array([[2.0, 1.0], [0.5, 3.0], [1.5, 0.2], [0.1, 2.2]])
        t = np.array([0, 1, 0, 0])

        precision_value = funcs.evaluation.multi_precision(y, t, target_class=1)
        recall_value = funcs.evaluation.multi_recall(y, t, target_class=1)
        f_value = funcs.evaluation.multi_f_score(y, t, target_class=1)

        for value in (precision_value, recall_value, f_value):
            self.assertGreaterEqual(float(value.data), 0.0)
            self.assertLessEqual(float(value.data), 1.0)


class TestRandomForestReproducibility(unittest.TestCase):
    """The same seed should reproduce the same forest (trees now get per-tree seeds)."""

    def test_same_seed_same_predictions(self):
        np.random.seed(2)
        x = np.random.randn(80, 3)
        t = (x[:, 0] + x[:, 1] > 0).astype(int)

        first = marquetry.ml.RandomForest(n_trees=5, max_depth=4, seed=11)
        first.fit(x, t)
        second = marquetry.ml.RandomForest(n_trees=5, max_depth=4, seed=11)
        second.fit(x, t)

        np.testing.assert_array_equal(first.predict(x).data, second.predict(x).data)

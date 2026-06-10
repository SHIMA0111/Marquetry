import os
import tempfile
import unittest

import numpy as np

import marquetry
import marquetry.functions as funcs
import marquetry.layers as layers
from marquetry.utils import array_close, array_equal


class TestLinearLayer(unittest.TestCase):

    def test_lazy_init_infers_in_size(self):
        linear = layers.Linear(7)
        x = np.random.randn(4, 3).astype(np.float32)

        y = linear(x)

        self.assertEqual(y.shape, (4, 7))
        self.assertEqual(linear.w.shape, (3, 7))
        self.assertEqual(linear.in_size, 3)

    def test_nobias(self):
        linear = layers.Linear(5, nobias=True)
        x = np.random.randn(2, 3).astype(np.float32)
        linear(x)

        self.assertIsNone(linear.b)
        self.assertEqual(len(list(linear.params())), 1)

    def test_bias_param_count(self):
        linear = layers.Linear(5)
        linear(np.random.randn(2, 3).astype(np.float32))

        self.assertEqual(len(list(linear.params())), 2)

    def test_output_matches_manual_computation(self):
        linear = layers.Linear(4)
        x = np.random.randn(3, 6).astype(np.float32)

        y = linear(x)
        expected = x.dot(linear.w.data) + linear.b.data

        self.assertTrue(array_close(y.data, expected))


class TestConvolution2DLayer(unittest.TestCase):

    def test_lazy_init_and_shape(self):
        conv = layers.Convolution2D(8, (3, 3), stride=1, pad=1)
        x = np.random.randn(2, 3, 5, 5).astype(np.float32)

        y = conv(x)

        self.assertEqual(y.shape, (2, 8, 5, 5))
        self.assertEqual(conv.w.shape, (8, 3, 3, 3))

    def test_stride_changes_output_size(self):
        conv = layers.Convolution2D(4, (3, 3), stride=2, pad=0)
        x = np.random.randn(1, 2, 9, 9).astype(np.float32)

        y = conv(x)

        self.assertEqual(y.shape, (1, 4, 4, 4))


class TestDeconvolution2DLayer(unittest.TestCase):

    def test_shape(self):
        deconv = layers.Deconvolution2D(4, (2, 2), stride=2, pad=0)
        x = np.random.randn(2, 3, 4, 4).astype(np.float32)

        y = deconv(x)

        # out = stride * (in - 1) + kernel - 2 * pad = 2 * 3 + 2 = 8
        self.assertEqual(y.shape, (2, 4, 8, 8))


class TestLayerNormalizationLayer(unittest.TestCase):

    def test_normalizes_each_row(self):
        layer_norm = layers.LayerNormalization()
        x = np.random.randn(4, 6).astype(np.float32) * 3.0 + 1.0

        y = layer_norm(x)

        self.assertEqual(y.shape, x.shape)
        np.testing.assert_allclose(y.data.mean(axis=1), np.zeros(4), atol=1e-4)


class TestParamManagement(unittest.TestCase):

    def test_clear_grads(self):
        linear = layers.Linear(3)
        x = np.random.randn(2, 4).astype(np.float32)

        y = funcs.sum(linear(x))
        y.backward()
        self.assertIsNotNone(linear.w.grad)

        linear.clear_grads()
        self.assertIsNone(linear.w.grad)
        self.assertIsNone(linear.b.grad)

    def test_nested_layer_params(self):
        model = marquetry.models.MLP([4, 3, 2], is_dropout=False)
        model(np.random.randn(2, 5).astype(np.float32))

        # 3 Linear layers, each with w and b
        self.assertEqual(len(list(model.params())), 6)

    def test_save_load_roundtrip(self):
        x = np.random.randn(4, 5).astype(np.float32)

        model = marquetry.models.MLP([6, 4, 2], is_dropout=False)
        y_before = model(x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "mlp.npz")
            model.save_params(path)

            restored = marquetry.models.MLP([6, 4, 2], is_dropout=False)
            restored.load_params(path)

        y_after = restored(x)
        self.assertTrue(array_equal(y_before.data, y_after.data))

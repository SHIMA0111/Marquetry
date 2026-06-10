import unittest

import numpy as np

import marquetry
import marquetry.functions as funcs
import marquetry.layers as layers
from marquetry.models import MLP, Sequential, CNN
from marquetry.utils import array_close


class TestMLP(unittest.TestCase):

    def test_forward_shape(self):
        model = MLP([8, 4], is_dropout=False)
        x = np.random.randn(5, 3).astype(np.float32)

        y = model(x)

        self.assertEqual(y.shape, (5, 4))

    def test_training_reduces_loss(self):
        np.random.seed(0)
        x = np.random.randn(64, 3).astype(np.float32)
        true_w = np.random.randn(3, 1).astype(np.float32)
        t = marquetry.Container(x.dot(true_w))

        model = MLP([8, 1], activation=funcs.relu, is_dropout=False)
        optimizer = marquetry.optimizers.SGD(lr=0.1).prepare(model)

        initial_loss = None
        final_loss = None
        for _ in range(200):
            model.clear_grads()
            loss = funcs.mean_squared_error(model(x), t)
            loss.backward()
            optimizer.update()

            if initial_loss is None:
                initial_loss = float(loss.data)
            final_loss = float(loss.data)

        self.assertLess(final_loss, initial_loss * 0.5)


class TestSequential(unittest.TestCase):

    def test_equals_manual_chain(self):
        model = Sequential(layers.Linear(4), funcs.relu, layers.Linear(2))
        x = np.random.randn(3, 5).astype(np.float32)

        y = model(x)
        expected = model.layers[2](funcs.relu(model.layers[0](x)))

        self.assertTrue(array_close(y.data, expected.data))

    def test_accepts_layer_list(self):
        model = Sequential([layers.Linear(4), layers.Linear(2)])
        x = np.random.randn(3, 5).astype(np.float32)

        y = model(x)

        self.assertEqual(y.shape, (3, 2))


class TestCNN(unittest.TestCase):

    def test_forward_shape(self):
        model = CNN(out_size=4)
        x = np.random.randn(2, 1, 12, 12).astype(np.float32)

        with marquetry.test_mode():
            y = model(x)

        self.assertEqual(y.shape, (2, 4))

    def test_backward_reaches_all_params(self):
        model = CNN(out_size=3)
        x = np.random.randn(2, 1, 12, 12).astype(np.float32)

        loss = funcs.sum(model(x))
        loss.backward()

        grads = [param.grad for param in model.params()]
        self.assertTrue(all(grad is not None for grad in grads))

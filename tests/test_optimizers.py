import unittest

import numpy as np
import torch

import marquetry
import marquetry.functions as funcs
from marquetry import Container
from marquetry.utils import array_close


def run_optimization(optimizer, iterations=200):
    """Minimize mean_squared_error(p, target) from p=0 and report (initial, final) loss."""
    layer = marquetry.Layer()
    layer.p = marquetry.Parameter(np.zeros(5))
    target = Container(np.full(5, 2.0))

    optimizer.prepare(layer)

    initial_loss = None
    for _ in range(iterations):
        layer.clear_grads()
        loss = funcs.mean_squared_error(layer.p, target)
        loss.backward()

        if initial_loss is None:
            initial_loss = float(loss.data)

        optimizer.update()

    final_loss = float(funcs.mean_squared_error(layer.p, target).data)

    return initial_loss, final_loss


class TestSGD(unittest.TestCase):

    def test_matches_torch(self):
        w0 = np.random.randn(4, 3).astype(np.float32)

        layer = marquetry.Layer()
        layer.p = marquetry.Parameter(w0.copy())
        optimizer = marquetry.optimizers.SGD(lr=0.1).prepare(layer)

        torch_param = torch.nn.Parameter(torch.tensor(w0))
        torch_optimizer = torch.optim.SGD([torch_param], lr=0.1)

        for _ in range(3):
            grad = np.random.randn(4, 3).astype(np.float32)
            layer.p.grad = Container(grad.copy())
            optimizer.update()

            torch_param.grad = torch.tensor(grad)
            torch_optimizer.step()

        self.assertTrue(array_close(layer.p.data, torch_param.detach().numpy()))

    def test_convergence(self):
        initial, final = run_optimization(marquetry.optimizers.SGD(lr=0.3))
        self.assertLess(final, initial * 0.1)


class TestMomentumSGD(unittest.TestCase):

    def test_convergence(self):
        initial, final = run_optimization(marquetry.optimizers.MomentumSGD(lr=0.1))
        self.assertLess(final, initial * 0.1)


class TestNesterov(unittest.TestCase):

    def test_convergence(self):
        initial, final = run_optimization(marquetry.optimizers.Nesterov(lr=0.1))
        self.assertLess(final, initial * 0.1)


class TestAdaGrad(unittest.TestCase):

    def test_convergence(self):
        initial, final = run_optimization(marquetry.optimizers.AdaGrad(lr=0.5))
        self.assertLess(final, initial * 0.1)


class TestAdaDelta(unittest.TestCase):

    def test_convergence(self):
        # AdaDelta self-tunes its step size and starts very slowly,
        # so only a monotonic improvement is asserted here.
        initial, final = run_optimization(marquetry.optimizers.AdaDelta(), iterations=500)
        self.assertLess(final, initial)


class TestRMSProp(unittest.TestCase):

    def test_convergence(self):
        initial, final = run_optimization(marquetry.optimizers.RMSProp(lr=0.05))
        self.assertLess(final, initial * 0.1)


class TestAdam(unittest.TestCase):

    def test_convergence(self):
        initial, final = run_optimization(marquetry.optimizers.Adam(lr=0.1))
        self.assertLess(final, initial * 0.1)


class TestAdamW(unittest.TestCase):

    def test_convergence(self):
        initial, final = run_optimization(marquetry.optimizers.AdamW(lr=0.1))
        self.assertLess(final, initial * 0.1)


class TestAdaMax(unittest.TestCase):

    def test_convergence(self):
        initial, final = run_optimization(marquetry.optimizers.AdaMax(lr=0.1))
        self.assertLess(final, initial * 0.1)


class TestNadam(unittest.TestCase):

    def test_convergence(self):
        initial, final = run_optimization(marquetry.optimizers.Nadam(lr=0.1))
        self.assertLess(final, initial * 0.1)


class TestLion(unittest.TestCase):

    def test_convergence(self):
        initial, final = run_optimization(marquetry.optimizers.Lion(lr=0.01), iterations=300)
        self.assertLess(final, initial * 0.1)


class TestWeightDecayHook(unittest.TestCase):

    def test_decay_pulls_param_to_zero(self):
        layer = marquetry.Layer()
        layer.p = marquetry.Parameter(np.ones(3))
        optimizer = marquetry.optimizers.SGD(lr=0.5).prepare(layer)
        optimizer.add_hook(marquetry.WeightDecay(0.1))

        layer.p.grad = Container(np.zeros(3))
        optimizer.update()

        # p_new = p - lr * (0 + rate * p) = 1 - 0.5 * 0.1
        self.assertTrue(array_close(layer.p.data, np.full(3, 0.95)))


class TestClipGradHook(unittest.TestCase):

    def test_update_norm_is_clipped(self):
        layer = marquetry.Layer()
        layer.p = marquetry.Parameter(np.zeros(2))
        optimizer = marquetry.optimizers.SGD(lr=1.0).prepare(layer)
        optimizer.add_hook(marquetry.ClipGrad(1.0))

        layer.p.grad = Container(np.array([3.0, 4.0]))
        optimizer.update()

        update_norm = float(np.sqrt((layer.p.data ** 2).sum()))
        self.assertLessEqual(update_norm, 1.0 + 1e-6)
        # direction is preserved: 3:4 ratio
        self.assertTrue(array_close(layer.p.data / update_norm, np.array([-0.6, -0.8])))

import unittest

import numpy as np
import torch

import marquetry
import marquetry.functions as funcs
from marquetry.utils import gradient_check, array_close


class TestSigmoid(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        y = funcs.sigmoid(x)
        ty = torch.nn.functional.sigmoid(torch.tensor(x))

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_forward2(self):
        x = np.random.randn(10, 10).astype(np.float32)
        y = funcs.sigmoid(x)
        ty = torch.nn.functional.sigmoid(torch.tensor(x))

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])

        self.assertTrue(gradient_check(funcs.sigmoid, x_data))

    def test_backward2(self):
        x_data = np.random.rand(10, 10)

        self.assertTrue(gradient_check(funcs.sigmoid, x_data))

    def test_backward3(self):
        x_data = np.random.randn(10, 10, 10)

        self.assertTrue(gradient_check(funcs.sigmoid, x_data))


class TestReLU(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.relu(x)

        res = y.data
        expected = np.array([[0, 0], [2, 0], [0, 1]], np.float32)

        self.assertTrue(array_close(res, expected))

    def test_forward2(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.relu(x)
        ty = torch.relu(torch.tensor(x))

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])

        self.assertTrue(gradient_check(funcs.relu, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10) * 100

        self.assertTrue(gradient_check(funcs.relu, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 10, 10) * 100

        self.assertTrue(gradient_check(funcs.relu, x_data))


class TestSoftmax(unittest.TestCase):
    def test_forward1(self):
        x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        y = funcs.softmax(x)
        ty = torch.softmax(torch.tensor(x), dim=1)

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_forward2(self):
        x = np.random.rand(10, 10).astype("f")
        y = funcs.softmax(x)
        ty = torch.softmax(torch.tensor(x), dim=1)

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_forward3(self):
        x = np.random.randn(10, 10, 10).astype("f")
        y = funcs.softmax(x, axis=2)
        ty = torch.softmax(torch.tensor(x), dim=2)

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])
        f = lambda x: funcs.softmax(x, axis=1)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)
        f = lambda x: funcs.softmax(x, axis=1)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 10, 10)
        f = lambda x: funcs.softmax(x, axis=2)

        self.assertTrue(gradient_check(f, x_data))


class TestLogSoftmax(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], dtype=np.float32)
        y = funcs.log_softmax(x)
        ty = torch.log_softmax(torch.tensor(x), dim=1, dtype=torch.float32)

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x_data = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]])
        f = lambda x: funcs.log_softmax(x)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)
        f = lambda x: funcs.log_softmax(x)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 10, 10)
        f = lambda x: funcs.log_softmax(x, axis=2)

        self.assertTrue(gradient_check(f, x_data))


class TestLeakyReLU(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.leaky_relu(x)

        res = y.data
        expected = np.array([[-0.2, 0], [2, -0.6], [-0.4, 1]], np.float32)

        self.assertTrue(array_close(res, expected))

    def test_forward2(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.leaky_relu(x)
        ty = torch.nn.functional.leaky_relu(torch.tensor(x), negative_slope=0.2)

        self.assertTrue(array_close(y.data, ty.data.numpy()))

    def test_backward1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])

        self.assertTrue(gradient_check(funcs.leaky_relu, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)

        self.assertTrue(gradient_check(funcs.leaky_relu, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 10, 10) * 100

        self.assertTrue(gradient_check(funcs.leaky_relu, x_data))


class TestGELU(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.gelu(x)

        expected = torch.nn.GELU()(torch.tensor(x))

        self.assertTrue(array_close(y.data, expected.numpy()))

    def test_forward2(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.gelu(x, approximate="tanh")

        expected = torch.nn.GELU(approximate="tanh")(torch.tensor(x))

        self.assertTrue(array_close(y.data, expected.numpy()))

    def test_forward3(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.gelu(x, approximate="sigmoid")

        expected = x * funcs.sigmoid(1.702 * x)

        self.assertTrue(array_close(y.data, expected.data))

    def test_backward1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])

        self.assertTrue(gradient_check(funcs.gelu, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)

        self.assertTrue(gradient_check(funcs.gelu, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 10)

        f = lambda x: funcs.gelu(x, approximate="tanh")

        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.random.rand(10, 10)

        f = lambda x: funcs.gelu(x, approximate="sigmoid")

        self.assertTrue(gradient_check(f, x_data))


class TestGLU(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.glu(x)

        expected = torch.nn.GLU()(torch.tensor(x))

        self.assertTrue(array_close(y.data, expected.numpy()))

    def test_forward2(self):
        x = np.random.randn(20, 3)
        y = funcs.glu(x, axis=0)

        expected = torch.nn.GLU(dim=0)(torch.tensor(x))

        self.assertTrue(array_close(y.data, expected.numpy()))

    def test_exception1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])

        with self.assertRaises(RuntimeError):
            funcs.glu(x_data)

    def test_backward1(self):
        x_data = np.random.randn(10, 10)

        self.assertTrue(gradient_check(funcs.glu, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 10)

        f = lambda x: funcs.glu(x, axis=0)

        self.assertTrue(gradient_check(f, x_data))


class TestPReLU(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        alpha = np.array(0.25)
        y = funcs.prelu(x, alpha)

        expected = torch.nn.PReLU()(torch.tensor(x))

        self.assertTrue(array_close(y.data, expected.detach().numpy()))

    def test_forward2(self):
        x = np.random.randn(20, 3)
        alpha = np.array([0.12, 0.32, 0.55])
        y = funcs.prelu(x, alpha)

        torch_prelu = torch.nn.PReLU(3)
        torch_prelu.weight = torch.nn.Parameter(torch.tensor(alpha))
        expected = torch_prelu(torch.tensor(x))

        self.assertTrue(array_close(y.data, expected.detach().numpy()))

    def test_exception1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])

        with self.assertRaises(RuntimeError):
            funcs.prelu(x_data, np.array([0.12, 0.12]))

    def test_backward1(self):
        x_data = np.random.randn(10, 10)
        alpha = np.array([0.55])

        f = lambda x: funcs.prelu(x, alpha)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)
        alpha = np.random.randn(10)

        f = lambda x: funcs.prelu(x, alpha)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.randn(10, 10)
        alpha = np.array([0.55])

        f = lambda alpha: funcs.prelu(x_data, alpha)

        self.assertTrue(gradient_check(f, alpha))

    def test_backward4(self):
        x_data = np.random.randn(10, 10)
        alpha = np.random.randn(10)

        f = lambda alpha: funcs.prelu(x_data, alpha)

        self.assertTrue(gradient_check(f, alpha))


class TestSoftPlus(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], dtype=np.float32)
        y = funcs.softplus(x)
        ty = torch.nn.Softplus()(torch.tensor(x))

        self.assertTrue(array_close(y.data, ty.detach().numpy()))

    def test_forward2(self):
        x = np.array([[100, -1000, 503, 781], [50, 100, 25, -1]], dtype=np.float32)
        y = funcs.softplus(x)
        ty = torch.nn.Softplus()(torch.tensor(x))
        self.assertTrue(array_close(y.data, ty.detach().numpy()))

    def test_forward3(self):
        x = np.array([[100, -1000, 503, 781], [50, 100, 25, -1]], dtype=np.float32)
        y = funcs.softplus(x, beta=2)
        ty = torch.nn.Softplus(beta=2)(torch.tensor(x))
        self.assertTrue(array_close(y.data, ty.detach().numpy()))

    def test_backward1(self):
        x_data = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]])

        self.assertTrue(gradient_check(funcs.softplus, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)

        self.assertTrue(gradient_check(funcs.softplus, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 10)
        f = lambda x: funcs.softplus(x, beta=3)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.array([[100, -1000, 503, 781], [50, 100, 25, -1]])

        self.assertTrue(gradient_check(funcs.softplus, x_data))


class TestSwish(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], dtype=np.float32)
        y = funcs.swish(x)
        ty = torch.nn.SiLU()(torch.tensor(x))

        self.assertTrue(array_close(y.data, ty.detach().numpy()))

    def test_forward2(self):
        x = np.array([[100, -1000, 503, 781], [50, 100, 25, -1]], dtype=np.float32)
        y = funcs.swish(x)
        ty = torch.nn.SiLU()(torch.tensor(x))
        self.assertTrue(array_close(y.data, ty.detach().numpy()))

    def test_forward3(self):
        x = np.array([[100, -1000, 503, 781], [50, 100, 25, -1]], dtype=np.float32)
        y = funcs.swish(x, beta=2)
        expected = x * funcs.sigmoid(2 * x)
        self.assertTrue(array_close(y.data, expected.data))

    def test_forward4(self):
        x = np.array([[100, -1000, 503, 781], [50, 100, 25, -1]], dtype=np.float32)
        y = funcs.dynamic_swish(x, beta=2)
        expected = x * funcs.sigmoid(2 * x)
        self.assertTrue(array_close(y.data, expected.data))

    def test_forward5(self):
        x = np.array([[100, -1000, 503, 781], [50, 100, 25, -1]], dtype=np.float32)
        pswish = marquetry.layers.DynamicSwish()
        y = pswish(x)
        expected = x * funcs.sigmoid(pswish.beta.data * x)
        self.assertTrue(array_close(y.data, expected.data))

    def test_backward1(self):
        x_data = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]])

        self.assertTrue(gradient_check(funcs.swish, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 10)

        self.assertTrue(gradient_check(funcs.swish, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 10)
        f = lambda x: funcs.swish(x, beta=3)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.array([[100, -1000, 503, 781], [50, 100, 25, -1]])

        f = lambda x: funcs.dynamic_swish(x, beta=1)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward5(self):
        x_data = np.random.rand(10, 10)
        beta = np.array(0.25)
        f = lambda beta: funcs.dynamic_swish(x_data, beta=beta)

        self.assertTrue(gradient_check(f, beta))

    def test_backward6(self):
        x_data = np.random.rand(10, 10)
        pswish = marquetry.layers.DynamicSwish()

        self.assertTrue(gradient_check(pswish, x_data))


class TestMish(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        y = funcs.mish(x)

        expected = torch.nn.Mish()(torch.tensor(x))

        self.assertTrue(array_close(y.data, expected.detach().numpy()))

    def test_forward2(self):
        x = np.random.randn(20, 3)
        y = funcs.mish(x)

        expected = torch.nn.Mish()(torch.tensor(x))

        self.assertTrue(array_close(y.data, expected.detach().numpy()))

    def test_backward1(self):
        x_data = np.random.randn(10, 10)

        self.assertTrue(gradient_check(funcs.mish, x_data))

    def test_backward3(self):
        x_data = np.random.rand(5, 5, 5)

        self.assertTrue(gradient_check(funcs.mish, x_data))


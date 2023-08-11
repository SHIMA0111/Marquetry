import unittest

import numpy as np

import marquetry.functions as funcs
from marquetry import Variable
from marquetry.utils import array_close, array_equal, gradient_check


class TestReshape(unittest.TestCase):

    def test_shape_check1(self):
        x = Variable(np.random.randn(1, 9))
        y = x.reshape((3, 3))

        expected_shape = (3, 3)

        self.assertTrue(y.shape, expected_shape)

    def test_shape_check2(self):
        x = Variable(np.random.randn(1, 16))
        y = x.reshape((4, 4))

        y.backward()

        self.assertTrue(array_equal(x.grad.shape, x.shape))


class TestTranspose(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = funcs.transpose(x)

        self.assertEqual(y.shape, (3, 2))

    def test_forward2(self):
        x = Variable(np.random.randn(2, 3, 4))
        y = funcs.transpose(x, (1, 0, 2))

        self.assertEqual(y.shape, (3, 2, 4))

    def test_backward1(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])

        self.assertTrue(gradient_check(funcs.transpose, x))

    def test_backward2(self):
        x = np.array([1, 2, 3])

        self.assertTrue(gradient_check(funcs.transpose, x))

    def test_backward3(self):
        x = np.random.randn(10, 5)

        self.assertTrue(gradient_check(funcs.transpose, x))

    def test_backward4(self):
        x = np.array([1, 2])

        self.assertTrue(gradient_check(funcs.transpose, x))

    def test_backward5(self):
        x = np.random.randn(3, 4, 5)

        self.assertTrue(gradient_check(funcs.transpose, x, (1, 0, 2)))


class TestGetItem(unittest.TestCase):

    def test_forward1(self):
        x_data = np.arange(12).reshape((2, 2, 3))
        x = Variable(x_data)
        y = x[0]

        self.assertTrue(array_close(y.data, x_data[0]))

    def test_forward2(self):
        x_data = np.arange(18).reshape((3, 3, 2))
        x = Variable(x_data)
        y = funcs.get_item(x, 0)

        self.assertTrue(array_close(y.data, x_data[0]))

    def test_forward3(self):
        x_data = np.arange(27).reshape((3, 3, 3))
        x = Variable(x_data)
        y = funcs.get_item(x, (Ellipsis, 2))

        self.assertTrue(array_close(y.data, x_data[..., 2]))

    def test_forward4(self):
        x_data = np.arange(36).reshape((3, 3, 4))
        x = Variable(x_data)
        y = funcs.get_item(x, (0, 1, slice(0, 2, 2)))

        self.assertTrue(array_close(y.data, x_data[0, 1, 0:2:2]))

    def test_backward1(self):
        x_data = np.array([[1, 2, 3], [4, 5, 6]])
        slices = 1
        f = lambda x: funcs.get_item(x, slices)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.arange(32).reshape((8, 4))
        slices = slice(2, 3)
        f = lambda x: funcs.get_item(x, slices)

        self.assertTrue(gradient_check(f, x_data))


class TestRepeat(unittest.TestCase):
    def test_check_shape1(self):
        x = Variable(np.random.randn(3, 1, 5))
        y = x.repeat(5, axis=1)

        self.assertEqual(y.shape, (3, 5, 5))

    def test_check_shape2(self):
        x = Variable(np.random.randn(3, 1, 5))
        y = funcs.repeat(x, repeats=4, axis=2)

        self.assertEqual(y.shape, (3, 1, 20))

    def test_forward1(self):
        x = Variable(np.arange(12).reshape((3, 4)))
        y = x.repeat(2, axis=0)

        self.assertTrue(array_close(y.data[0::2, :], x.data))
        self.assertTrue(array_close(y.data[1::2, :], x.data))

    def test_backward1(self):
        x = np.arange(24).reshape((4, 6))
        repeats = 3
        axis = 0

        f = lambda x: funcs.repeat(x, repeats, axis)

        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = np.random.randn(3, 1, 4)
        repeats = 5
        axis = 1

        f = lambda x: x.repeat(repeats, axis)

        self.assertTrue(gradient_check(f, x))

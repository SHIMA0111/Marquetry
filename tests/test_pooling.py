import unittest

import numpy as np
import torch
import torch.nn.functional as F

import marquetry.functions as funcs
from marquetry.utils import array_close, gradient_check


class TestMaxPooling2D(unittest.TestCase):

    def test_forward_matches_torch(self):
        x = np.random.randn(2, 3, 8, 8).astype(np.float32)

        y = funcs.max_pooling_2d(x, kernel_size=(2, 2), stride=2)
        expected = F.max_pool2d(torch.tensor(x), kernel_size=2, stride=2)

        self.assertTrue(array_close(y.data, expected.numpy()))

    def test_forward_kernel3_stride1(self):
        x = np.random.randn(1, 2, 6, 6).astype(np.float32)

        y = funcs.max_pooling_2d(x, kernel_size=(3, 3), stride=1)
        expected = F.max_pool2d(torch.tensor(x), kernel_size=3, stride=1)

        self.assertTrue(array_close(y.data, expected.numpy()))

    def test_backward(self):
        x = np.random.randn(2, 2, 6, 6)
        f = lambda x_in: funcs.max_pooling_2d(x_in, kernel_size=(2, 2), stride=2)

        self.assertTrue(gradient_check(f, x))

    def test_backward_overlapping_windows(self):
        x = np.random.randn(1, 2, 5, 5)
        f = lambda x_in: funcs.max_pooling_2d(x_in, kernel_size=(2, 2), stride=1)

        self.assertTrue(gradient_check(f, x))

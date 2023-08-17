import unittest

import numpy as np
import torch

import marquetry.functions as funcs
from marquetry import Variable
from marquetry.utils import gradient_check, array_close, array_equal


class TestConv2d(unittest.TestCase):

    def test_forward1(self):
        batch_size, channels, height, width = 1, 5, 15, 15
        output_channels, kernel_size, stride, padding = 8, (3, 3), (1, 1), (1, 1)
        x = np.random.randn(batch_size, channels, height, width).astype("f")
        w = np.random.randn(output_channels, channels, kernel_size[0], kernel_size[1]).astype("f")
        b = None
        tb = torch.Tensor(b) if b is not None else b
        excepted = torch.conv2d(torch.Tensor(x), torch.Tensor(w), tb, stride, padding)
        y = funcs.conv2d(x, w, b, stride, padding)

        self.assertTrue(array_close(excepted.data.numpy(), y.data))

    def test_forward2(self):
        batch_size, channels, height, width = 1, 5, 15, 15
        output_channels, kernel_size, stride, padding = 8, (3, 3), (3, 1), (2, 1)
        x = np.random.randn(batch_size, channels, height, width).astype("f")
        w = np.random.randn(output_channels, channels, kernel_size[0], kernel_size[1]).astype("f")
        b = None
        tb = torch.Tensor(b) if b is not None else b
        expected = torch.conv2d(torch.Tensor(x), torch.Tensor(w), tb, stride, padding)
        y = funcs.conv2d(x, w, b, stride, padding)

        self.assertTrue(array_close(expected.data.numpy(), y.data))

    def test_forward3(self):
        batch_size, channels, height, width = 1, 5, 20, 15
        output_channels, kernel_size, stride, padding = 8, (5, 3), 1, 3
        x = np.random.randn(batch_size, channels, height, width).astype("f")
        w = np.random.randn(output_channels, channels, kernel_size[0], kernel_size[1]).astype("f")
        b = None
        tb = torch.Tensor(b) if b is not None else b
        expected = torch.conv2d(torch.Tensor(x), torch.Tensor(w), tb, stride, padding)
        y = funcs.conv2d(x, w, b, stride, padding)

        self.assertTrue(array_close(expected.data.numpy(), y.data))

    def test_forward4(self):
        batch_size, channels, height, width = 1, 5, 20, 15
        output_channels, kernel_size, stride, padding = 8, (5, 3), 1, 3
        x = np.random.randn(batch_size, channels, height, width).astype("f")
        w = np.random.randn(output_channels, channels, kernel_size[0], kernel_size[1]).astype("f")
        b = np.random.randn(output_channels).astype("f")
        tb = torch.Tensor(b) if b is not None else b
        expected = torch.conv2d(torch.Tensor(x), torch.Tensor(w), tb, stride, padding)
        y = funcs.conv2d(x, w, b, stride, padding)

        self.assertTrue(array_close(expected.data.numpy(), y.data))

    def test_backward1(self):
        batch_size, channels, height, width = 1, 5, 20, 15
        output_channels, kernel_size, stride, padding = 3, (5, 3), 1, 3
        x = np.random.randn(batch_size, channels, height, width)
        w = np.random.randn(output_channels, channels, kernel_size[0], kernel_size[1]).astype("f")
        b = np.random.randn(output_channels)

        f = lambda x: funcs.conv2d(x, w, b, stride, padding)

        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        batch_size, channels, height, width = 1, 5, 20, 15
        output_channels, kernel_size, stride, padding = 3, (5, 3), 1, 3
        x = np.random.randn(batch_size, channels, height, width)
        w = np.random.randn(output_channels, channels, kernel_size[0], kernel_size[1]).astype("f")
        b = np.random.randn(output_channels)

        f = lambda b: funcs.conv2d(x, w, b, stride, padding)

        self.assertTrue(gradient_check(f, b))

    def test_backward3(self):
        batch_size, channels, height, width = 1, 5, 20, 15
        output_channels, kernel_size, stride, padding = 3, (5, 3), 1, 3
        x = np.random.randn(batch_size, channels, height, width)
        w = np.random.randn(output_channels, channels, kernel_size[0], kernel_size[1])
        b = np.random.randn(output_channels)

        f = lambda w: funcs.conv2d(x, w, b, stride, padding)

        self.assertTrue(gradient_check(f, w))


class TestDeconv2d(unittest.TestCase):

    def test_forward1(self):
        batch_size, channel_input, channel_output = 10, 1, 3
        height_input, width_input = 5, 10
        height_kernel, width_kernel = 10, 10
        height_padding, width_padding = 5, 5
        stride_y, stride_x = 5, 5

        x = np.random.uniform(0, 1, (batch_size, channel_input, height_input, width_input)).astype("f")
        w = np.random.uniform(0, 1, (channel_input, channel_output, height_padding, width_kernel)).astype("f")
        b = np.random.uniform(0, 1, channel_output).astype("f")

        expected = torch.conv_transpose2d(torch.Tensor(x), torch.Tensor(w), torch.Tensor(b), (stride_y, stride_x), (height_padding, width_padding))
        y = funcs.deconv2d(x, w, b, (stride_y, stride_x), (height_padding, width_padding))

        self.assertTrue(array_close(expected.data.numpy(), y.data))

    def test_forward2(self):
        batch_size, channel_input, channel_output = 10, 1, 3
        height_input, width_input = 5, 10
        height_kernel, width_kernel = 10, 10
        height_padding, width_padding = 5, 5
        stride_y, stride_x = 5, 5

        x = np.random.uniform(0, 1, (batch_size, channel_input, height_input, width_input)).astype("f")
        w = np.random.uniform(0, 1, (channel_input, channel_output, height_padding, width_kernel)).astype("f")
        b = None

        expected = torch.conv_transpose2d(torch.Tensor(x), torch.Tensor(w), b, (stride_y, stride_x),
                                          (height_padding, width_padding))
        y = funcs.deconv2d(x, w, b, (stride_y, stride_x), (height_padding, width_padding))

        self.assertTrue(array_close(expected.data.numpy(), y.data))

    def test_backward1(self):
        batch_size, channel_input, channel_output = 10, 1, 3
        height_input, width_input = 5, 10
        height_kernel, width_kernel = 10, 10
        height_padding, width_padding = 5, 5
        stride_y, stride_x = 5, 5

        x = np.random.uniform(0, 1, (batch_size, channel_input, height_input, width_input))
        w = np.random.uniform(0, 1, (channel_input, channel_output, height_padding, width_kernel))
        b = None

        f = lambda x: funcs.deconv2d(x, w, b, stride=(stride_y, stride_x), pad=(height_padding, width_padding))

        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        batch_size, channel_input, channel_output = 10, 1, 3
        height_input, width_input = 5, 10
        height_kernel, width_kernel = 10, 10
        height_padding, width_padding = 5, 5
        stride_y, stride_x = 5, 5

        x = np.random.uniform(0, 1, (batch_size, channel_input, height_input, width_input))
        w = np.random.uniform(0, 1, (channel_input, channel_output, height_padding, width_kernel))
        b = np.random.uniform(0, 1, channel_output)

        f = lambda w: funcs.deconv2d(x, w, b, stride=(stride_y, stride_x), pad=(height_padding, width_padding))

        self.assertTrue(gradient_check(f, w))

    def test_backward3(self):
        batch_size, channel_input, channel_output = 10, 1, 3
        height_input, width_input = 5, 10
        height_kernel, width_kernel = 10, 10
        height_padding, width_padding = 5, 5
        stride_y, stride_x = 5, 5

        x = np.random.uniform(0, 1, (batch_size, channel_input, height_input, width_input))
        w = np.random.uniform(0, 1, (channel_input, channel_output, height_padding, width_kernel))
        b = np.random.uniform(0, 1, channel_output)

        f = lambda b: funcs.deconv2d(x, w, b, stride=(stride_y, stride_x), pad=(height_padding, width_padding))

        self.assertTrue(gradient_check(f, b))

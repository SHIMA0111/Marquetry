import sys

import numpy as np

import marquetry
from marquetry import utils, cuda_backend
from marquetry.core import Function, as_variable, as_array


# ===========================================================================
# Basic functions: sin / cos / tanh / exp / log
# ===========================================================================
class Sin(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.sin(x)

        return y

    def backward(self, x, grad_y):
        grad_x = cos(x[0]) * grad_y[0]

        return grad_x


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.cos(x)

        return y

    def backward(self, x, grad_y):
        grad_x = -sin(x[0]) * grad_y[0]

        return grad_x


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.tanh(x)

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = grad_y[0] * (1 - y ** 2)

        return grad_x


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.exp(x)

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = grad_y[0] * y

        return grad_x


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.log(x)

        return y

    def backward(self, x, grad_y):
        grad_x = grad_y[0] / x[0]

        return grad_x


def log(x):
    return Log()(x)


# ===========================================================================
# Tensor operations: reshape / transpose / get_item / repeat /
#                    concat / split / squeeze / unsqueeze / flatten
# ===========================================================================
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        return reshape(grad_y[0], self.x_shape)


def reshape(x, shape):
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        if self.axes is None:
            return transpose(grad_y[0])

        xp = cuda_backend.get_array_module(grad_y[0])

        axes_len = len(self.axes)
        inv_axes = tuple(xp.argsort([ax % axes_len for ax in self.axes]))
        return transpose(grad_y[0], inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]

        return y

    def backward(self, x, grad_y):
        f = GetItemGrad(self.slices, x[0].shape)
        return f(grad_y[0])


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, grad_y):
        xp = cuda_backend.get_array_module(grad_y)
        grad_x = xp.zeros(self.in_shape, dtype=grad_y.dtype)

        if xp is np:
            np.add.at(grad_x, self.slices, grad_y)
        else:
            cuda_backend.cpx.scatter_add(grad_x, self.slices, grad_y)

        self.retain_inputs(())
        return grad_x

    def backward(self, x, grad_grad_y):
        return get_item(grad_grad_y[0], self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


class Repeat(Function):
    def __init__(self, repeat_num, axis):
        self.repeat_num = repeat_num
        self.axis = axis

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.repeat(x, self.repeat_num, self.axis)

        return y

    def backward(self, x, grad_y):
        x_shape = x[0].shape

        grad_x = RepeatGrad(x_shape, self.repeat_num, self.axis)(grad_y[0])

        return grad_x


class RepeatGrad(Function):
    def __init__(self, in_shape, repeat_num, axis):
        self.in_shape = in_shape
        self.repeat_num = repeat_num
        self.axis = axis

    def forward(self, grad_y):
        xp = cuda_backend.get_array_module(grad_y)

        original_num = self.in_shape[self.axis]
        grad_shape = list(grad_y.shape)
        grad_shape[self.axis - 1] *= original_num
        grad_shape[self.axis] = int(grad_shape[self.axis] / original_num)
        grad_shape = tuple(grad_shape)

        grad_y = grad_y.reshape(grad_shape)
        grad_y = xp.sum(grad_y, axis=self.axis)
        grad_x = grad_y.reshape(self.in_shape)

        self.retain_inputs(())
        return grad_x

    def backward(self, x, grad_grad_y):
        grad_grad_x = repeat(grad_grad_y[0], self.repeat_num, self.axis)

        return grad_grad_x


def repeat(x, repeats, axis):
    return Repeat(repeats, axis)(x)


class Concat(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            inputs = tuple(inputs[0])

        xp = cuda_backend.get_array_module(inputs[0])
        y = xp.concatenate(inputs, axis=self.axis)

        return y

    def backward(self, inputs, grad_y):
        pre_index = 0
        indices = []
        for i, data in enumerate(inputs):
            if i == len(inputs) - 1:
                continue
            index = data.shape[self.axis]
            pre_index += index
            indices.append(pre_index)

        grad_x = split(grad_y[0], indices, axis=self.axis)

        return grad_x


def concat(*inputs, axis=0):
    if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
        inputs = tuple(inputs[0])

    return Concat(axis)(*inputs)


class Split(Function):
    def __init__(self, indices, axis):
        self.axis = axis
        if np.isscalar(indices):
            indices = (indices,)
        self.indices = indices

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.split(x, self.indices, axis=self.axis)

        self.retain_inputs(())
        return tuple(y)

    def backward(self, x, grad_ys):
        grad_x = concat(grad_ys, axis=self.axis)

        return grad_x


def split(x, indices, axis):
    return Split(indices, axis)(x)


class Squeeze(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        if x.shape[self.axis] != 1:
            raise ValueError("You can't squeeze non-one size axis element.")

        y = xp.squeeze(x, axis=self.axis)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        grad_x = unsqueeze(grad_y[0], self.axis)

        return grad_x


def squeeze(x, axis):
    return Squeeze(axis)(x)


class UnSqueeze(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        x_shape = x.shape

        new_shape = list(x_shape)
        new_shape.insert(self.axis, 1)
        new_shape = tuple(new_shape)

        y = x.reshape(new_shape)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        grad_x = squeeze(grad_y[0], self.axis)

        return grad_x


def unsqueeze(x, axis):
    return UnSqueeze(axis)(x)


def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    return reshape(x, (x.shape[0], -1))


# ===========================================================================
# Tensor calc: sum / sum_to / broadcast_to / average / matmul / linear
# ===========================================================================
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        grad_y = utils.reshape_sum_backward(grad_y[0], self.x_shape, self.axis, self.keepdims)
        grad_x = broadcast_to(grad_y[0], self.x_shape)
        return grad_x


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

        self.x_shape = None

    def forward(self, x):
        if x.shape == self.shape:
            return x

        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        if self.x_shape is None:
            return grad_y[0]

        grad_x = broadcast_to(grad_y[0], self.x_shape)

        return grad_x


def sum_to(x, shape):
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

        self.x_shape = None

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        if x.shape == self.shape:
            return x

        self.x_shape = x.shape
        y = xp.broadcast_to(x, self.shape)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        if self.x_shape is None:
            return grad_y[0]

        grad_x = sum_to(grad_y[0], self.x_shape)

        return grad_x


def broadcast_to(x, shape):
    return BroadcastTo(shape)(x)


def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = sum(x, axis, keepdims)

    return y * (y.data.size / x.data.size)


mean = average


class MatMul(Function):
    def forward(self, x1, x2):
        y = x1.dot(x2)

        return y

    def backward(self, xs, grad_y):
        x1, x2 = xs
        grad_y = grad_y[0]

        grad_x1 = matmul(grad_y, x2.T)
        grad_x2 = matmul(x1.T, grad_y)

        return grad_x1, grad_x2


def matmul(x1, x2):
    return MatMul()(x1, x2)


class Linear(Function):
    def forward(self, x, w, b):
        y = x.dot(w)
        if b is not None:
            y += b
        return y

    def backward(self, inputs, grad_y):
        x, w, b = inputs
        grad_y = grad_y[0]

        grad_b = None if b is None else sum_to(grad_y, b.shape)

        grad_x = matmul(grad_y, w.T)
        grad_w = matmul(x.T, grad_y)

        return grad_x, grad_w, grad_b


def linear(x, w, b=None):
    return Linear()(x, w, b)


# ===========================================================================
# Convolution: conv2d / deconv2d / pooling / average_pooling
# ===========================================================================
class Conv2D(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = utils.pair(stride)
        self.pad = utils.pair(pad)

    def forward(self, x, w, b):
        xp = cuda_backend.get_array_module(x)

        kernel_height, kernel_width = w.shape[2:]
        col = utils.im2col_array(x, (kernel_height, kernel_width), self.stride, self.pad, to_matrix=False)

        y = xp.tensordot(col, w, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b

        y = xp.rollaxis(y, 3, 1)

        return y

    def backward(self, inputs, grad_y):
        x, w, b = inputs
        grad_y = grad_y[0]

        grad_x = deconv2d(grad_y, w, b=None, stride=self.stride, pad=self.pad, out_size=(x.shape[2], x.shape[3]))

        grad_w = Conv2DGradW(self)(x, grad_y)

        grad_b = None

        if b is not None:
            grad_b = grad_y.sum(axis=(0, 2, 3))

        return grad_x, grad_w, grad_b


def conv2d(x, w, b=None, stride=1, pad=0):
    return Conv2D(stride, pad)(x, w, b)


class Deconv2D(Function):
    def __init__(self, stride=1, pad=0, out_size=None):
        super().__init__()
        self.stride = utils.pair(stride)
        self.pad = utils.pair(pad)
        self.out_size = out_size

        self.no_bias = False

    def forward(self, x, w, b):
        xp = cuda_backend.get_array_module(x)

        stride_height, stride_width = self.stride
        padding_height, padding_width = self.pad
        channels, out_channels, kernel_height, kernel_width = w.shape

        batch_size, channels, height, width = x.shape

        if self.out_size is None:
            out_height = utils.get_deconv_outsize(height, kernel_height, stride_height, padding_height)
            out_width = utils.get_deconv_outsize(width, kernel_width, stride_width, padding_width)
        else:
            out_height, out_width = utils.pair(self.out_size)

        img_shape = (batch_size, out_channels, out_height, out_width)
        grad_col = xp.tensordot(w, x, (0, 1))
        grad_col = xp.rollaxis(grad_col, 3)

        y = utils.col2im_array(
            grad_col, img_shape, (kernel_height, kernel_width), self.stride, self.pad, to_matrix=False)

        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))

        return y

    def backward(self, inputs, grad_y):
        x, w, b = inputs
        grad_y = grad_y[0]

        grad_x = conv2d(grad_y, w, b=None, stride=self.stride, pad=self.pad)

        grad_w = Conv2DGradW(self)(grad_y, x)

        grad_b = None
        if b is not None:
            grad_b = grad_y.sum(axis=(0, 2, 3))

        return grad_x, grad_w, grad_b


def deconv2d(x, w, b=None, stride=1, pad=0, out_size=None):
    return Deconv2D(stride, pad, out_size=out_size)(x, w, b)


class Conv2DGradW(Function):
    def __init__(self, conv2d_instance):
        w = conv2d_instance.inputs[1]
        kernel_height, kernel_width = w.shape[2:]
        self.kernel_size = (kernel_height, kernel_width)
        self.stride = conv2d_instance.stride
        self.pad = conv2d_instance.pad

    def forward(self, x, grad_y):
        xp = cuda_backend.get_array_module(x)

        col = utils.im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        grad_w = xp.tensordot(grad_y, col, ((0, 2, 3), (0, 4, 5)))

        self.retain_outputs((0,))
        return grad_w

    def backward(self, inputs, grad_ys):
        x, grad_y = inputs
        grad_w, = self.output_data

        x_height, x_width = x.shape[2:]
        grad_x = deconv2d(grad_y, grad_w, stride=self.stride, pad=self.pad, out_size=(x_height, x_width))
        grad_grad_y = conv2d(x, grad_w, stride=self.stride, pad=self.pad)

        return grad_x, grad_grad_y


class MaxPooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_shape = None
        self.input_dtype = None

        self.indexes = None

    def forward(self, x):
        self.input_shape = x.shape
        self.input_dtype = x.dtype

        col = utils.im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        batch_size, channels, kernel_height, kernel_weight, out_height, out_width = col.shape
        col = col.reshape((batch_size, channels, kernel_height * kernel_weight, out_height, out_width))

        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)

        self.retain_inputs(())
        return y

    def backward(self, x, grad_y):
        return MaxPooling2DGrad(self)(grad_y[0])


class MaxPooling2DGrad(Function):
    def __init__(self, pooling2d):
        self.pooling2d = pooling2d
        self.kernel_size = pooling2d.kernel_size
        self.stride = pooling2d.stride
        self.pad = pooling2d.pad
        self.input_shape = pooling2d.input_shape
        self.dtype = pooling2d.input_dtype
        self.indexes = pooling2d.indexes

        self.shape = None

    def forward(self, grad_y):
        self.shape = grad_y.shape
        self.dtype = grad_y.dtype

        xp = cuda_backend.get_array_module(grad_y)

        batch_size, channels, output_height, output_width = grad_y.shape
        batch_size, channels, height, width = self.input_shape
        kernel_height, kernel_width = utils.pair(self.kernel_size)

        grad_col = xp.zeros(
            (batch_size * channels * output_height * output_width * kernel_height * kernel_width), dtype=self.dtype)

        indexes = (self.indexes.ravel() + xp.arange(
            0, self.indexes.size * kernel_height * kernel_width, kernel_height * kernel_width))
        grad_col[indexes] = grad_y.ravel()
        grad_col = grad_col.reshape((batch_size, channels, output_height, output_width, kernel_height, kernel_width))
        grad_col = xp.swapaxes(grad_col, 2, 4)
        grad_col = xp.swapaxes(grad_col, 3, 5)

        grad_x = utils.col2im_array(grad_col, (batch_size, channels, height, width),
                                    self.kernel_size, self.stride, self.pad, to_matrix=False)

        self.retain_inputs(())
        return grad_x

    def backward(self, x, grad_grad_y):
        f = Pooling2DWithIndexes(self)
        return f(grad_grad_y[0])


class Pooling2DWithIndexes(Function):
    def __init__(self, pooling2d):
        self.kernel_size = pooling2d.kernel_size
        self.stride = pooling2d.stride
        self.pad = pooling2d.pad
        self.input_shape = pooling2d.shape
        self.dtype = pooling2d.dtype
        self.indexes = pooling2d.indexes

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        col = utils.im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        batch_size, channels, kernel_height, kernel_width, out_height, out_width = col.shape

        col = col.reshape((batch_size, channels, kernel_height * kernel_width, out_height, out_width))
        col = col.transpose((0, 1, 3, 4, 2)).reshape(-1, kernel_height * kernel_width)
        indexes = self.indexes.ravel()
        col = col[xp.arange(len(indexes)), indexes]

        self.retain_inputs(())
        return col.reshape(batch_size, channels, out_height, out_width)


def max_pool(x, kernel_size, stride=1, pad=0):
    return MaxPooling(kernel_size, stride, pad)(x)


# ===========================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# ===========================================================================
class Sigmoid(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        y = xp.exp(xp.minimum(0, x)) / (1 + xp.exp(-xp.abs(x)))

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = y * (1 - y) * grad_y[0]

        return grad_x


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        xp = cuda_backend.get_array_module(x)
        y = xp.maximum(x, 0.0)

        return y

    def backward(self, x, grad_y):
        x, = x
        mask = x > 0
        grad_x = grad_y[0] * mask

        return grad_x


def relu(x):
    return ReLU()(x)


class Softmax(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_x = y * grad_y[0]
        sum_grad_x = grad_x.sum(axis=self.axis, keepdims=True)
        grad_x -= y * sum_grad_x

        return grad_x


def softmax(x, axis=1):
    return Softmax(axis)(x)


class LogSoftmax(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z

        self.retain_inputs(())
        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        y = self.output_data[0]
        grad_y = grad_y[0]

        grad_x = grad_y - exp(y) * grad_y.sum(axis=self.axis, keepdims=True)

        return grad_x


def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)


class LeakyReLU(Function):
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x <= 0] *= self.slope

        return y

    def backward(self, x, grad_y):
        mask = (x[0] > 0).astype(grad_y.dtype)
        mask[mask <= 0] = self.slope

        grad_x = grad_y * mask

        return grad_x


def leaky_relu(x, slope=0.2):
    return LeakyReLU(slope)(x)


# ===========================================================================
# loss function: mean_squared_error / softmax_cross_entropy / sigmoid_cross_entropy / binary_cross_entropy
# ===========================================================================
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)

        return y

    def backward(self, inputs, grad_y):
        x0, x1 = inputs
        diff = x0 - x1
        grad_x0 = 2. * diff / len(diff) * grad_y[0]
        grad_x1 = -grad_x0 * grad_y[0]

        return grad_x0, grad_x1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        xp = cuda_backend.get_array_module(x)
        batch_size = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[xp.arange(batch_size), t.ravel()]
        y = -log_p.sum() / xp.float32(batch_size)

        return y

    def backward(self, inputs, grad_y):
        x, t = inputs
        grad_y = grad_y[0]

        batch_size, data_dim = x.shape

        grad_y *= 1 / batch_size
        y = softmax(x)
        xp = cuda_backend.get_array_module(t)
        if y.size != t.size:
            # convert class num to one-hot
            t_onehot = xp.eye(data_dim, dtype=t.dtype)[t]
        else:
            t_onehot = t

        y = (y - t_onehot) * grad_y
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


class SigmoidCrossEntropy(Function):
    def __init__(self):
        self.batch_size = None

    def forward(self, x, t):
        if x.ndim != t.ndim:
            t = t.reshape(*x.shape)

        xp = cuda_backend.get_array_module(x)

        batch_size = x.shape[0] if x.ndim != 1 else len(x)
        p = xp.exp(xp.minimum(0, x)) / (1 + xp.exp(-xp.abs(x)))
        p = xp.clip(p, 1e-15, .999)
        tlog_p = t * xp.log(p) + (1 - t) * xp.log(1 - p)
        y = -1 * tlog_p.sum() / batch_size

        self.batch_size = batch_size

        return y

    def backward(self, inputs, grad_y):
        x, t = inputs
        if x.ndim != t.ndim:
            t = t.reshape(*x.shape)
        y = sigmoid(x)

        batch_size = self.batch_size

        # grad_x = -(1 / batch_size) * ((t / y) - ((1 - t) / (1 - y))) * (y * (1 - y)) * grad_y
        grad_x = -(1 / batch_size) * (t * (1 - y) - y * (1 - t)) * grad_y[0]

        return grad_x


def sigmoid_cross_entropy(x, t):
    return SigmoidCrossEntropy()(x, t)


def classification_cross_entropy(x, t):
    if x.ndim == 1 or x.shape[1] == 1:
        return sigmoid_cross_entropy(x, t)
    else:
        return softmax_cross_entropy(x, t)


def simple_sigmoid_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)

    x, t = as_variable(x), as_variable(t)
    batch_size = len(x)
    p = sigmoid(x)
    p = clip(p, 1e-15, 0.99)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / batch_size
    return y


def binary_cross_entropy(p, t):
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    batch_size = len(p)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / batch_size
    return y


# ===========================================================================
# dropout / batch_norm
# ===========================================================================
def accuracy(y, t, threshold=0.7):
    """
    The `threshold` affects only binary prediction so if you use multiple classification, the parameter will be ignored.
    """
    xp = cuda_backend.get_array_module(y.data)

    y, t = as_variable(y), as_variable(t)

    if y.ndim == 1:
        y = y.reshape((-1, 1))

    if y.shape[1] == 1:
        pred = (y.data >= threshold).astype(xp.int32).reshape(t.shape)
    else:
        pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return marquetry.Variable(as_array(acc))


class Dropout(Function):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

        self.mask = None

    def forward(self, x):
        if marquetry.Config.train_mode:
            xp = cuda_backend.get_array_module(x)
            mask = xp.random.rand(*x.shape) > self.dropout_rate
            self.mask = mask
            scale = xp.array(1.0 - self.dropout_rate).astype(x.dtype)
            y = x * mask / scale
        else:
            y = x

        self.retain_inputs(())

        return y

    def backward(self, x, grad_y):
        if marquetry.Config.train_mode:
            grad_x = grad_y[0] * self.mask
        else:
            raise Exception("You execute non-train mode so you can't do backward.")

        return grad_x


def dropout(x, dropout_rate=0.5):
    return Dropout(dropout_rate)(x)


class BatchNorm(Function):
    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps

        self.inv_std = None

    def forward(self, x, gamma, beta):
        assert x.ndim in (2, 4)

        x_ndim = x.ndim
        x_shape = x.shape
        if x_ndim == 4:
            batch_size, channels, height, width = x_shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, channels)

        xp = cuda_backend.get_array_module(x)

        if marquetry.Config.train_mode:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            normed_x = (x - mean) * inv_std

            samples = x.size // gamma.size
            scale = samples - 1. if samples - 1. > 1. else 1.
            adjust = samples / scale
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean

            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var

            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            normed_x = (x - self.avg_mean) * inv_std

        y = gamma * normed_x + beta

        if x_ndim == 4:
            batch_size, channels, height, width = x_shape
            y = y.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)

        self.retain_inputs((0, 1))
        return y

    def backward(self, inputs, grad_y):
        grad_y = grad_y[0]

        gy_ndim = grad_y.ndim
        gy_shape = grad_y.shape
        if gy_ndim == 4:
            batch_size, channels, height, width = gy_shape
            grad_y = grad_y.transpose(0, 2, 3, 1).reshape(-1, channels)

        x, gamma, _ = inputs
        batch_size = len(x)

        if x.ndim == 4:
            batch_size, channels, height, width = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, channels)

        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        grad_beta = sum(grad_y, axis=0)
        grad_gamma = sum(xc * grad_y, axis=0)

        grad_x = grad_y - grad_beta / batch_size - xc * grad_gamma / batch_size
        grad_x *= gamma * self.inv_std

        if gy_ndim == 4:
            batch_size, channels, height, width = gy_shape
            grad_x = grad_x.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)

        return grad_x, grad_gamma, grad_beta


def batch_norm(x, gamma, beta, mean, var, decay=0.9, eps=1e-15):
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)


# ===========================================================================
# Im2col / Col2im
# ===========================================================================
class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

        self.input_shape = None
        self.x_shape = None

    def forward(self, x):
        self.input_shape = x.shape

        y = utils.im2col_array(x, kernel_size=self.kernel_size, stride=self.stride, pad=self.pad, to_matrix=self.to_matrix)

        self.retain_inputs(())
        return y

    def backward(self, grad_y):
        grad_x = col2im(grad_y[0], self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

        return grad_x


def im2col(img, kernel_size, stride=1, pad=0, to_matrix=True):
    return Im2col(kernel_size, stride, pad, to_matrix)(img)


class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()

        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = utils.col2im_array(x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

        self.retain_inputs(())
        return y

    def backward(self, grad_y):
        grad_x = im2col(grad_y[0], self.kernel_size, self.stride, self.pad, self.to_matrix)

        return grad_x


def col2im(col, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(col)


# ===========================================================================
# max / min / clip
# ===========================================================================
class Max(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)

        self.retain_outputs((0,))
        return y

    def backward(self, x, grad_y):
        x = x[0]
        y = self.output_data[0]
        grad_y = grad_y[0]

        xp = cuda_backend.get_array_module(x)

        shape = utils.max_backward_shape(x, self.axis)
        grad_y = reshape(grad_y, shape)
        y = reshape(y, shape)
        cond = xp.array(x == y)
        grad_y = broadcast_to(grad_y, cond.shape)

        return grad_y * cond


class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda_backend.get_array_module(x)

        y = xp.clip(x, self.x_min, self.x_max)

        return y

    def backward(self, x, grad_y):
        x = x[0]

        mask = (x >= self.x_min) * (x <= self.x_max)
        grad_x = grad_y[0] * mask

        return grad_x


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

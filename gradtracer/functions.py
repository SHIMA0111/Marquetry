import numpy as np

import gradtracer
from gradtracer import utils
from gradtracer.core import Function, as_variable


# ===========================================================================
# Basic functions: sin / cos / tanh / exp / log
# ===========================================================================
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)

        return y

    def backward(self, grad_y):
        x, = self.inputs
        grad_x = cos(x) * grad_y

        return grad_x


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)

        return y

    def backward(self, grad_y):
        x, = self.inputs
        grad_x = -sin(x) * grad_y

        return grad_x


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)

        return y

    def backward(self, grad_y):
        y = self.outputs[0]()
        grad_x = grad_y * (1 - y ** 2)

        return grad_x


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)

        return y

    def backward(self, grad_y):
        y = self.outputs[0]()
        grad_x = grad_y * y

        return grad_x


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        y = np.log(x)

        return y

    def backward(self, grad_y):
        x, = self.inputs
        grad_x = grad_y / x

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

        return y

    def backward(self, grad_y):
        return reshape(grad_y, self.x_shape)


def reshape(x, shape):
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)

        return y

    def backward(self, grad_y):
        if self.axes is None:
            return transpose(grad_y)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(grad_y, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]

        return y

    def backward(self, grad_y):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(grad_y)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, grad_y):
        grad_x = np.zeros(self.in_shape, dtype=grad_y.dtype)
        np.add.at(grad_x, self.slices, grad_y)

        return grad_x

    def backward(self, grad_grad_y):
        return get_item(grad_grad_y, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


class Repeat(Function):
    def __init__(self, repeat_num, axis):
        self.repeat_num = repeat_num
        self.axis = axis

    def forward(self, x):
        y = np.repeat(x, self.repeat_num, self.axis)

        return y

    def backward(self, grad_y):
        x, = self.inputs
        x_shape = x.shape

        grad_x = RepeatGrad(x_shape, self.repeat_num, self.axis)(grad_y)

        return grad_x


class RepeatGrad(Function):
    def __init__(self, in_shape, repeat_num, axis):
        self.in_shape = in_shape
        self.repeat_num = repeat_num
        self.axis = axis

    def forward(self, grad_y):
        original_num = self.in_shape[self.axis]
        grad_shape = list(grad_y.shape)
        grad_shape[self.axis - 1] *= original_num
        grad_shape[self.axis] = int(grad_shape[self.axis] / original_num)
        grad_shape = tuple(grad_shape)

        grad_y = grad_y.reshape(grad_shape)
        grad_y = np.sum(grad_y, axis=self.axis)
        grad_x = grad_y.reshape(self.in_shape)

        return grad_x

    def backward(self, grad_grad_y):
        grad_grad_x = repeat(grad_grad_y, self.repeat_num, self.axis)

        return grad_grad_x


def repeat(x, repeats, axis):
    return Repeat(repeats, axis)(x)


class Concat(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            inputs = tuple(inputs[0])

        y = np.concatenate(inputs, axis=self.axis)

        return y

    def backward(self, grad_y):
        inputs = self.inputs

        pre_index = 0
        indices = []
        for i, data in enumerate(inputs):
            if i == len(inputs) - 1:
                continue
            index = data.shape[self.axis]
            pre_index += index
            indices.append(pre_index)

        grad_x = split(grad_y, indices, axis=self.axis)

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
        y = np.split(x, self.indices, axis=self.axis)

        return tuple(y)

    def backward(self, *grad_ys):
        grad_x = concat(grad_ys, axis=self.axis)

        return grad_x


def split(x, indices, axis):
    return Split(indices, axis)(x)


class Squeeze(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        if x.shape[self.axis] != 1:
            raise ValueError("You can't squeeze non-one size axis element.")

        y = np.squeeze(x, axis=self.axis)
        return y

    def backward(self, grad_y):
        grad_x = unsqueeze(grad_y, self.axis)

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

        y = np.reshape(x, new_shape)

        return y

    def backward(self, grad_y):
        grad_x = squeeze(grad_y, self.axis)

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
        return y

    def backward(self, grad_y):
        grad_y = utils.reshape_sum_backward(grad_y, self.x_shape, self.axis, self.keepdims)
        grad_x = broadcast_to(grad_y, self.x_shape)
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

        return y

    def backward(self, grad_y):
        if self.x_shape is None:
            return grad_y

        grad_x = broadcast_to(grad_y, self.x_shape)

        return grad_x


def sum_to(x, shape):
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

        self.x_shape = None

    def forward(self, x):
        if x.shape == self.shape:
            return x

        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)

        return y

    def backward(self, grad_y):
        if self.x_shape is None:
            return grad_y

        grad_x = sum_to(grad_y, self.x_shape)

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

    def backward(self, grad_y):
        x1, x2 = self.inputs
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

    def backward(self, grad_y):
        x, w, b = self.inputs
        grad_b = None if b.data is None else sum_to(grad_y, b.shape)

        grad_x = matmul(grad_y, w.T)
        grad_w = matmul(x.T, grad_y)

        return grad_x, grad_w, grad_b


def linear(x, w, b=None):
    return Linear()(x, w, b)


# ===========================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# ===========================================================================
class Sigmoid(Function):
    def forward(self, x):
        y = np.exp(np.minimum(0, x)) / (1 + np.exp(-np.abs(x)))

        return y

    def backward(self, grad_y):
        y = self.outputs[0]()
        grad_x = y * (1 - y) * grad_y

        return grad_x


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)

        return y

    def backward(self, grad_y):
        x, = self.inputs
        mask = x.data > 0
        grad_x = grad_y * mask

        return grad_x


def relu(x):
    return ReLU()(x)


class Softmax(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)

        return y

    def backward(self, grad_y):
        y = self.outputs[0]()
        grad_x = y * grad_y
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

        return y

    def backward(self, grad_y):
        y = self.outputs[0]()
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

    def backward(self, grad_y):
        x, = self.inputs
        mask = (x.data > 0).astype(grad_y.dtype)
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

    def backward(self, grad_y):
        x0, x1 = self.inputs
        diff = x0 - x1
        grad_x0 = 2. * diff / len(diff)
        grad_x1 = -grad_x0

        return grad_x0, grad_x1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        batch_size = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(batch_size), t.ravel()]
        y = -log_p.sum() / np.float32(batch_size)

        return y

    def backward(self, grad_y):
        x, t = self.inputs
        batch_size, data_dim = x.shape

        grad_y *= 1 / batch_size
        y = softmax(x)
        if y.size != t.size:
            # convert class num to one-hot
            t_onehot = np.eye(data_dim, dtype=t.dtype)[t.data]
        else:
            t_onehot = t

        y = (y - t_onehot) * grad_y
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


# class SigmoidCrossEntropy(Function):
#     def forward(self, x, t):
#         if x.ndim != t.ndim:
#             t = t.reshape(*x.shape)
#
#         batch_size = len(x)
#         p = np.exp(np.minimum(0, x)) / (1 + np.exp(-np.abs(x)))
#         p = np.clip(p, 1e-15, 1.)
#         tlog_p = t * np.log(p) + (1 - t) * np.log(1 - p)
#         y = -1 * tlog_p.sum() / batch_size
#
#         return y
#
#     def backward(self, grad_y):
#         x, t = self.inputs
#         if x.ndim != t.ndim:
#             t = t.reshape(*x.shape)
#         y = sigmoid(x)


def sigmoid_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)

    x, t = as_variable(x), as_variable(t)
    batch_size = len(x)
    p = sigmoid(x)
    p = clip(p, 1e-15, 1.0)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / batch_size
    return y


def binary_cross_entropy(p, t):
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    batch_size = len(p)
    p = clip(p, 1e-15, 1.0)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / batch_size
    return y


# ===========================================================================
# dropout / batch_norm
# ===========================================================================
class Dropout(Function):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

        self.mask = None

    def forward(self, x):
        if gradtracer.Config.train_mode:
            mask = np.random.rand(*x.shape) > self.dropout_rate
            self.mask = mask
            scale = np.array(1.0 - self.dropout_rate).astype(x.dtype)
            y = x * mask / scale
        else:
            y = x

        return y

    def backward(self, grad_y):
        if gradtracer.Config.train_mode:
            grad_x = grad_y * self.mask
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
        assert x.ndim == 2

        if gradtracer.Config.train_mode:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / np.sqrt(var + self.eps)
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
            inv_std = 1 / np.sqrt(self.avg_var + self.eps)
            normed_x = (x - self.avg_mean) * inv_std

        y = gamma * normed_x + beta
        return y

    def backward(self, grad_y):
        x, gamma, beta = self.inputs
        batch_size = len(x)

        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        grad_beta = sum(grad_y, axis=0)
        grad_gamma = sum(xc * grad_y, axis=0)

        grad_x = grad_y - grad_beta / batch_size - xc * grad_gamma / batch_size
        grad_x *= gamma * self.inv_std

        return grad_x, grad_gamma, grad_beta


def batch_norm(x, gamma, beta, mean, var, decay=0.9, eps=1e-15):
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)


# ===========================================================================
# max / min / clip
# ===========================================================================
class Max(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)

        return y

    def backward(self, grad_y):
        x = self.inputs[0]
        y = self.outputs[0]()

        shape = utils.max_backward_shape(x, self.axis)
        grad_y = reshape(grad_y, shape)
        y = reshape(y, shape)
        cond = np.array(x.data == y.data)
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
        y = np.clip(x, self.x_min, self.x_max)

        return y

    def backward(self, grad_y):
        x, = self.inputs

        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        grad_x = grad_y * mask

        return grad_x


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


# ===========================================================================
# label_encoder
# ===========================================================================
def label_encoder(x):
    if x.dtype == float:
        return x
    unique_set = set(x)
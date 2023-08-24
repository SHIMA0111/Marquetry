import os.path
import weakref

import numpy as np

import marquetry.functions as funcs
from marquetry import utils, cuda_backend
from marquetry.core import Parameter


# ===========================================================================
# Layer base class
# ===========================================================================
class Layer(object):
    def __init__(self):
        self._params = set()

    def __setattr__(self, key, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(key)
        super().__setattr__(key, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            data = self.__dict__[name]

            if isinstance(data, Layer):
                yield from data.params()
            else:
                yield data

    def clear_grads(self):
        for param in self.params():
            param.clear_grad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            data = self.__dict__[name]
            key = parent_key  + "/" + name if parent_key else name

            if isinstance(data, Layer):
                data._flatten_params(params_dict, key)
            else:
                params_dict[key] = data

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


# ===========================================================================
# Linear
# ===========================================================================
class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.outsize = out_size
        self.dtype = dtype

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="bias")

        self.w = Parameter(None, name="weight")

    def _init_w(self, xp=np):
        in_size, out_size = self.in_size, self.outsize
        w_data = xp.random.randn(in_size, out_size).astype(self.dtype) * xp.sqrt(1 / in_size)
        self.w.data = w_data
        if self.b is not None and xp is not np:
            self.b.to_gpu()

    def forward(self, x):
        if self.w.data is None:
            self.in_size = x.shape[-1]
            xp = cuda_backend.get_array_module(x)
            self._init_w(xp=xp)
        y = funcs.linear(x, self.w, self.b)
        return y


class Conv2D(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, nobias=False,
                 dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

        self.w = Parameter(None, name="w")

    def _init_w(self, xp=np):
        channels, output_channels = self.in_channels, self.out_channels
        kernel_height, kernel_width = utils.pair(self.kernel_size)

        scale = xp.sqrt(1 / (channels * kernel_height * kernel_width))
        w_data = xp.random.randn(output_channels, channels, kernel_height, kernel_width).astype(self.dtype) * scale

        self.w.data = w_data

        if self.b is not None and xp is not np:
            self.b.to_gpu()

    def forward(self, x):
        if self.w.data is None:
            self.in_channels = x.shape[1]
            xp = cuda_backend.get_array_module(x)
            self._init_w(xp=xp)

        y = funcs.conv2d(x, self.w, self.b, self.stride, self.pad)
        return y


class Deconv2D(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, nobias=False,
                 dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

        self.w = Parameter(None, name="w")

    def _init_w(self, xp=np):
        channels, out_channels = self.in_channels, self.out_channels
        kernel_height, kernel_width = utils.pair(self.kernel_size)
        scale = xp.sqrt(1 / (channels * kernel_height * kernel_width))
        w_data = xp.random.randn(channels, out_channels, kernel_height, kernel_width).astype(self.dtype) * scale
        self.w.data = w_data

        if self.b is not None and xp is not np:
            self.b.to_gpu()

    def forward(self, x):
        if self.w.data is None:
            self.in_channels = x.shape[1]
            xp = cuda_backend.get_array_module(x)
            self._init_w(xp=xp)

        y = funcs.deconv2d(x, self.w, self.b, self.stride, self.pad)

        return y


# ===========================================================================
# Natural Language Process
# ===========================================================================
class EmbedID(Layer):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.w = Parameter(np.random.randn(vocab_size, embed_size))

    def __call__(self, x):
        y = self.w[x]

        return y


class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def set_state(self, h):
        self.h = h

    def forward(self, x):
        if self.h is None:
            new_hidden_state = funcs.tanh(self.x2h(x))
        else:
            new_hidden_state = funcs.tanh(self.x2h(x) + self.h2h(self.h))

        self.h = new_hidden_state

        return new_hidden_state


class LSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        self.hidden_size = hidden_size

        self.x2hs = Linear(3 * hidden_size, in_size=in_size)
        self.x2i = Linear(hidden_size, in_size=in_size)
        self.h2hs = Linear(3 * hidden_size, in_size=hidden_size, nobias=True)
        self.h2i = Linear(hidden_size, in_size=hidden_size, nobias=True)

        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            hs = funcs.sigmoid(self.x2hs(x))
            input_data = funcs.tanh(self.x2i(x))
        else:
            hs = funcs.sigmoid(self.x2hs(x) + self.h2hs(self.h))
            input_data = funcs.tanh(self.x2i(x) + self.h2i(self.h))

        forget_gate = hs[:, :self.hidden_size]
        input_gate = hs[:, self.hidden_size:2 * self.hidden_size]
        output_gate = hs[:, 2 * self.hidden_size:]

        if self.c is None:
            c_new = input_gate * input_data
        else:
            c_new = (forget_gate * self.c) + (input_gate * input_data)

        h_new = output_gate * funcs.tanh(c_new)

        self.h, self.c = h_new, c_new

        return h_new


class BiLSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.forward_lstm = LSTM(hidden_size, in_size=in_size)
        self.reverse_lstm = LSTM(hidden_size, in_size=in_size)

    def reset_state(self):
        self.forward_lstm.reset_state()
        self.reverse_lstm.reset_state()

    def forward(self, x):
        out1 = self.forward_lstm(x)
        out2 = self.reverse_lstm(x[:, ::-1])
        out2 = out2[:, ::-1]

        output = funcs.concat((out1, out2), axis=-1)

        return output


class GRU(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.hidden_size = hidden_size

        self.x2h = Linear(hidden_size, in_size=in_size)
        self.x2r = Linear(hidden_size, in_size=in_size)
        self.x2u = Linear(hidden_size, in_size=in_size)

        self.h2h = Linear(hidden_size, in_size=hidden_size, nobias=True)
        self.h2r = Linear(hidden_size, in_size=hidden_size, nobias=True)
        self.h2u = Linear(hidden_size, in_size=hidden_size, nobias=True)

        self.h = None

    def reset_state(self):
        self.h = None

    def set_state(self, h):
        self.h = h

    def forward(self, x):
        if self.h is None:
            new_h = funcs.tanh(self.x2h(x))

        else:
            reset_gate = funcs.sigmoid(self.x2r(x) + self.h2r(self.h))
            new_h = funcs.tanh(self.x2h(x) + self.h2h(reset_gate * self.h))
            update_gate = funcs.sigmoid(self.x2u(x) + self.h2u(self.h))

            new_h = (1 - update_gate) * new_h + update_gate * self.h

        self.h = new_h

        return new_h


# ===========================================================================
# BatchNorm
# ===========================================================================
class BatchNorm(Layer):
    def __init__(self, decay=0.9):
        super().__init__()
        self.avg_mean = Parameter(None, name="avg_mean")
        self.avg_var = Parameter(None, name="avg_var")
        self.gamma = Parameter(None, name="gamma")
        self.beta = Parameter(None, name="beta")

        self.decay = decay

    def __call__(self, x):
        xp = cuda_backend.get_array_module(x)
        if self.avg_mean.data is None:
            input_shape = x.shape[1]
            if self.avg_mean.data is None:
                self.avg_mean.data = xp.zeros(input_shape, dtype=x.dtype)
            if self.avg_var.data is None:
                self.avg_var.data = xp.ones(input_shape, dtype=x.dtype)
            if self.gamma.data is None:
                self.gamma.data = xp.ones(input_shape, dtype=x.dtype)
            if self.beta.data is None:
                self.beta.data = xp.zeros(input_shape, dtype=x.dtype)

        return funcs.batch_norm(x, self.gamma, self.beta, self.avg_mean.data, self.avg_var.data, self.decay)

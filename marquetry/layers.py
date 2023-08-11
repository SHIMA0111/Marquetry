import os.path
import weakref

import numpy as np

import marquetry.functions as funcs
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

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            data = self.__dict__[name]
            key = parent_key  + "/" + name if parent_key else name

            if isinstance(data, Layer):
                data._flatten_params(params_dict, key)
            else:
                params_dict[key] = data

    def save_weights(self, path):
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

        self.w = Parameter(None, name="weight")
        if self.in_size is not None:
            self._init_w()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="bias")

    def _init_w(self):
        in_size, out_size = self.in_size, self.outsize
        w_data = np.random.randn(in_size, out_size).astype(self.dtype) * np.sqrt(1 / in_size)
        self.w.data = w_data

    def forward(self, x):
        if self.w.data is None:
            self.in_size = x.shape[-1]
            self._init_w()

        y = funcs.linear(x, self.w, self.b)
        return y


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
        if self.avg_mean.data is None:
            input_shape = x.shape[1]
            if self.avg_mean.data is None:
                self.avg_mean.data = np.zeros(input_shape, dtype=x.dtype)
            if self.avg_var.data is None:
                self.avg_var.data = np.ones(input_shape, dtype=x.dtype)
            if self.gamma.data is None:
                self.gamma.data = np.ones(input_shape, dtype=x.dtype)
            if self.beta.data is None:
                self.beta.data = np.zeros(input_shape, dtype=x.dtype)

        return funcs.batch_norm(x, self.gamma, self.beta, self.avg_mean.data, self.avg_var.data, self.decay)

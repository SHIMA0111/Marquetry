import os.path
import weakref

import numpy as np

from marquetry.variable import Parameter


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

    def save_params(self, path):
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

    def load_params(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]

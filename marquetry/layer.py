import os.path
import weakref

import numpy as np

from marquetry.container import Parameter


# ===========================================================================
# Layer base class
# ===========================================================================
class Layer(object):
    """Base class for neural network layers.

        The Layer class provides basic functionality for all layers in a neural network.
        Layers have a 'forward' method that wraps functions as a layer in the neural network.
        This class also stores parameters such as weights and biases,
        and the 'params' method returns these parameters.

        Additionally, this class includes methods for saving and loading layer weights,
        which allows you to persist and restore models.

    """
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
        """Perform the forward computation of the layer. Subclasses must implement this method.

            Args:
                *inputs (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                    Input data arrays.

            Returns:
                marquetry.Container: Output data arrays.

            Note:
                Generally, this class shouldn't be called by manually because `forward` is called by `__call__`.
        """

        raise NotImplementedError()

    def params(self):
        """Get parameters as an iterator.

            Yields:
                marquetry.Parameter: Layer parameters.
        """
        for name in self._params:
            data = self.__dict__[name]

            if isinstance(data, Layer):
                yield from data.params()
            else:
                yield data

    def clear_grads(self):
        """Clear gradients for all layer parameters."""
        for param in self.params():
            param.clear_grad()

    def to_cpu(self):
        """Move all layer parameters to the CPU."""
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        """Move all layer parameters to the GPU."""
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=""):
        """Helper function to flatten and organize layer parameters for saving.

            Args:
                params_dict (dict): Dictionary to store flattened parameters.
                parent_key (str): Key prefix for organizing parameters.
        """
        for name in self._params:
            data = self.__dict__[name]
            key = parent_key  + "/" + name if parent_key else name

            if isinstance(data, Layer):
                data._flatten_params(params_dict, key)
            else:
                params_dict[key] = data

    def save_params(self, path) -> None:
        """Save the layer's parameters to a specified file path.

            Args:
                path (str): The path where parameters will be saved.
        """
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

    def load_params(self, path) -> None:
        """Load previously saved parameters from a specified file path.

            Args:
                path (str): The path from which parameters will be loaded.
        """
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]

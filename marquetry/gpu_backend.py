import numpy as np

from marquetry import Variable


try:
    gpu_enable = True
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False


def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np

    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x

    return cp.asnumpy(x)


def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception("CuPy cannot be loaded. If you use Cuda environment, please install CuPy!")

    return cp.asarray(x)

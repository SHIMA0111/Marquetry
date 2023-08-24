import numpy as np

from marquetry import Variable

try:
    import cupy as cp
    import cupyx as cpx
    GPU_ENABLE = True

except ImportError:
    cp = np
    GPU_ENABLE = False


def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data

    if not GPU_ENABLE:
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

    if not GPU_ENABLE:
        raise Exception("CuPy cannot be loaded. Install CuPy at first.")
    return cp.asarray(x)

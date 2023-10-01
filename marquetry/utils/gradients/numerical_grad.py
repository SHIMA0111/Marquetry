from typing import Sequence

import numpy as np

from marquetry import cuda_backend
from marquetry import Container


def numerical_grad(func, x, *args, **kwargs):
    """Numerical Gradient calculator"""
    eps = 1e-4

    x = x.data if isinstance(x, Container) else x
    xp = cuda_backend.get_array_module(x)
    if xp is not np:
        np_x = cuda_backend.as_numpy(x)
    else:
        np_x = x

    grad = xp.zeros_like(x)

    iters = np.nditer(np_x, flags=["multi_index"], op_flags=["readwrite"])
    while not iters.finished:
        index = iters.multi_index
        tmp_val = x[index].copy()

        x[index] = tmp_val + eps
        y1 = func(x, *args, **kwargs)
        if isinstance(y1, Container):
            y1 = y1.data

        y1 = y1.copy()

        x[index] = tmp_val - eps
        y2 = func(x, *args, **kwargs)
        if isinstance(y2, Container):
            y2 = y2.data

        y2 = y2.copy()

        if isinstance(y1, list):
            diff = 0
            for i in range(len(y1)):
                diff += (y1[i] - y2[i]).sum()
        else:
            diff = (y1 - y2).sum()

        if isinstance(diff, Container):
            diff = diff.data

        grad[index] = diff / (2 * eps)

        x[index] = tmp_val
        iters.iternext()

    return grad

import numpy as np

from marquetry import cuda_backend
from marquetry import Variable


def array_equal(a, b):
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b

    a, b = cuda_backend.as_numpy(a), cuda_backend.as_numpy(b)

    return np.array_equal(a, b)
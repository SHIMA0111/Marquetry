import numpy as np

from marquetry import cuda_backend
from marquetry import Variable


def array_close(a, b, rtol=1e-4, atol=1e-5):
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b

    a, b = cuda_backend.as_numpy(a), cuda_backend.as_numpy(b)

    return np.allclose(a, b, rtol, atol)
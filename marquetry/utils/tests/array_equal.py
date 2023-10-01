import numpy as np

from marquetry import cuda_backend
from marquetry import Container


def array_equal(a, b):
    """Check if the arrays are the same or not."""
    a = a.data if isinstance(a, Container) else a
    b = b.data if isinstance(b, Container) else b

    a, b = cuda_backend.as_numpy(a), cuda_backend.as_numpy(b)

    return np.array_equal(a, b)
import marquetry
from marquetry import as_variable
from marquetry import as_array
from marquetry import cuda_backend


def accuracy(y, t, threshold=0.7):
    """
    The `threshold` affects only binary prediction so if you use multiple classification, the parameter will be ignored.
    """
    xp = cuda_backend.get_array_module(y.data)

    y, t = as_variable(y), as_variable(t)

    if y.ndim == 1:
        y = y.reshape((-1, 1))

    if y.shape[1] == 1:
        pred = (y.data >= threshold).astype(xp.int32).reshape(t.shape)
    else:
        pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return marquetry.Variable(as_array(acc))

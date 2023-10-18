from marquetry import cuda_backend


def sigmoid_array(x):
    xp = cuda_backend.get_array_module(x)

    y = xp.exp(xp.minimum(0, x)) / (1 + xp.exp(-xp.abs(x)))
    return y

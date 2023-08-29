from marquetry import cuda_backend


def logsumexp(x, axis=1):
    xp = cuda_backend.get_array_module(x)
    x_max = x.max(axis=axis, keepdims=True)
    y = x - x_max
    xp.exp(y, out=y)
    sum_exp = y.sum(axis=axis, keepdims=True)
    xp.log(sum_exp, out=sum_exp)
    y = x_max + sum_exp

    return y

from marquetry import as_variable, functions


def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = functions.sum(x, axis, keepdims)

    return y * (y.data.size / x.data.size)


mean = average

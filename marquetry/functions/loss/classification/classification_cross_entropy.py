from marquetry import functions


def classification_cross_entropy(x, t):
    if x.ndim == 1 or x.shape[1] == 1:
        return functions.sigmoid_cross_entropy(x, t)
    else:
        return functions.softmax_cross_entropy(x, t)

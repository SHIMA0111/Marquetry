from marquetry import functions


def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    return functions.reshape(x, (x.shape[0], -1))
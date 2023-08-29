def pair(x):
    if isinstance(x, int):
        return x, x
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError("pair can't use {}".format(x))
def pair(x):
    """Get a pair of ints.

        This function takes an input and returns a pair of values.
        If the input is an integer, it creates a pair of identical values.
        If the input is already a pair of values (tuple of length 2), it returns the input as is.

        For other types of input, a ValueError is raised.

        Args:
            x (int or tuple): The input value to create a pair from.

        Returns:
            tuple: A pair of ints.
    """

    if isinstance(x, int):
        return x, x
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError("pair can't use {}".format(x))

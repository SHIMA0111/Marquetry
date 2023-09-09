class Flatten(object):
    """A data transformation class for flattening multi-dimension arrays.

        Examples:
            >>> dataset = marquetry.datasets.MNIST(transform=Flatten())
    """

    def __call__(self, array):
        return array.flatten()

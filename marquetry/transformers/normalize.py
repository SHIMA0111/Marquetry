import numpy as np


class Normalize(object):
    """A data transformation class for normalizing data by subtracting mean and dividing by standard deviation.

        Examples:
            >>> mean = x.mean(axis=1)
            >>> std = x.std(axis=1)
            >>> dataset = marquetry.datasets.MNIST(transform=Normalize(mean=mean, std=std))

    """

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(mean) == 1 else len(mean)
            mean = np.array(mean, dtype=array.dtype).reshape(*mshape)

        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(std) == 1 else len(std)
            std = np.array(std, dtype=array.dtype).reshape(*rshape)

        return (array - mean) / std

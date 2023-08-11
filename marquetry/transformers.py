import numpy as np


# ===========================================================================
# Wrapper the transformer funcs
# ===========================================================================
class Compose(object):
    def __init__(self, transforms=None):
        self.transforms = transforms if len(transforms) != 0 else []

    def __call__(self, data):
        if not self.transforms:
            return data

        for transform_func in self.transforms:
            data = transform_func(data)

        return data


# ===========================================================================
# Transforms
# ===========================================================================
class Normalize(object):
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


class Flatten(object):
    def __call__(self, array):
        return array.flatten()


class AsType(object):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    def __init__(self, dtype=np.int32):
        super().__init__(dtype)

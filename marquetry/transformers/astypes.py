import numpy as np


class AsType(object):
    """A utility class for converting NumPy arrays to a specific data type.

        Args:
            dtype (numpy.dtype):
                The target data type to which arrays should be converted.

    """

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


class ToFloat(AsType):
    """
        A subclass of AsType for converting NumPy arrays to float data type.
        This class inherits from AsType and sets the target data type to float32.
    """

    def __init__(self):
        super().__init__(dtype=np.float32)


class ToInt(AsType):
    """
        A subclass of AsType for converting NumPy arrays to integer data type.
        This class inherits from AsType and sets the target data type to int32.
    """

    def __init__(self):
        super().__init__(dtype=np.int32)

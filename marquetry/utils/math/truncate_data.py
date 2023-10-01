import numpy as np

from marquetry import Container
from marquetry import config
from marquetry import cuda_backend


allowed_array = (np.ndarray,) if not config.CUDA_ENABLE else (np.ndarray, cuda_backend.cp.ndarray)


def floor(x):
    """Round elements of an input array down to the nearest integer.

        This function rounds each element in the input array down to the nearest integer.

        Args:
            x (:class:`Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array to be rounded.

        Returns:
            Container: A new Container object containing the rounded values.

        Raises:
            ValueError: Raised if the input is not a valid Container or ndarray.

        Example:
            >>> input_array = Container([2.7, 3.1, 4.9])
            >>> rounded_values = floor(input_array)
            >>> print(rounded_values)
            Container([2., 3., 4.])

        Note:
            - If the input is a Container, the name of the resulting Container will be inherited from the input.

        See Also:
            :func:`ceil`: Round elements up to the nearest integer.
    """

    name = None

    if isinstance(x, Container):
        if x.name is not None:
            name = x.name
        x = x.data
    elif not isinstance(x, allowed_array):
        raise ValueError("x expected `Container` or `ndarray`, but got {}.".format(type(x)))

    y = np.floor(x)

    return Container(y, name=name)


def ceil(x):
    """Round elements of an input array up to the nearest integer.

        This function rounds each element in the input array up to the nearest integer.

        Args:
            x (Container or ndarray): The input array to be rounded.

        Returns:
            Container: A new Container object containing the rounded values.

        Raises:
            ValueError: Raised if the input is not a valid Container or ndarray.

        Example:
            >>> input_array = Container([2.7, 3.1, 4.9])
            >>> rounded_values = ceil(input_array)
            >>> print(rounded_values)
            Container([3., 4., 5.])

        Note:
            - If the input is a Container, the name of the resulting Container will be inherited from the input.

        See Also:
            :func:`floor`: Round elements down to the nearest integer.
    """

    name = None

    if isinstance(x, Container):
        if x.name is not None:
            name = x.name
        x = x.data
    elif not isinstance(x, allowed_array):
        raise ValueError("x expected `Container` or `ndarray`, but got {}.".format(type(x)))

    y = np.ceil(x)

    return Container(y, name=name)

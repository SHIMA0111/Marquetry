import marquetry
from marquetry import as_array
from marquetry import Function


class Add(Function):
    """Compute element-wise addition of two arrays.

        This class computes the element-wise addition of two input arrays `x0` and `x1`.
        It supports broadcasting when the input shapes are not the same.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self):
        self.x0_shape = None
        self.x1_shape = None

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1

        self.retain_inputs(())

        return y

    def backward(self, x, grad_y):
        grad_x0, grad_x1 = grad_y[0], grad_y[0]
        if self.x0_shape != self.x1_shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, self.x1_shape)

        return grad_x0, grad_x1


def add(x0, x1):
    """Compute element-wise addition of two arrays.

        This function computes the element-wise addition of two input arrays `x0` and `x1`. \n
        It supports broadcasting when the input shapes are not the same.

        Args:
            x0 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The first input.
            x1 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The second input.

        Returns:
            :class:`marquetry.Container`: The result of element-wise addition.

        Examples:
            >>> x0 = np.array([1, 2, 3])
            >>> x1 = np.array([2, 3, 4])
            >>> add(x0, x1)
            container([3 5 7])

    """

    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
    """Compute element-wise multiplication of two arrays.

        This class computes the element-wise multiplication of two input arrays `x0` and `x1`.
        It supports broadcasting when the input shapes are not the same.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, inputs, grad_y):
        x0, x1 = inputs
        grad_x0 = grad_y[0] * x1
        grad_x1 = grad_y[0] * x0
        if x0.shape != x1.shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, x0.shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, x1.shape)

        return grad_x0, grad_x1


def mul(x0, x1):
    """Compute element-wise multiplication of two arrays.

        This function computes the element-wise multiplication of two input arrays `x0` and `x1`. \n
        It supports broadcasting when the input shapes are not the same.

        Args:
            x0 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The first input.
            x1 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The second input.

        Returns:
            :class:`marquetry.Container`: The result of element-wise multiplication.

        Examples:
            >>> x0 = np.array([1, 2, 3])
            >>> x1 = np.array([2, 3, 4])
            >>> mul(x0, x1)
            container([ 2  6 12])

    """

    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    """Compute element-wise negation arrays.

        This class computes the element-wise negation arrays `x`.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def forward(self, x):
        self.retain_inputs(())
        return -x

    def backward(self, x, grad_y):
        return -grad_y[0]


def neg(x):
    """Compute element-wise negation arrays.

        This function computes the element-wise negation arrays `x`.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array.

        Returns:
            :class:`marquetry.Container`: The result of element-wise negation.

        Examples:
            >>> x = np.array([1, 2, 3])
            >>> neg(x)
            container([-1 -2 -3])

    """

    return Neg()(x)


class Sub(Function):
    """Compute element-wise subtraction of two arrays.

        This class computes the element-wise subtraction of two input arrays `x0` and `x1`.
        It supports broadcasting when the input shapes are not the same.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self):
        self.x0_shape = None
        self.x1_shape = None

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1

        self.retain_inputs(())
        return y

    def backward(self, inputs, grad_x):
        grad_x0 = grad_x[0]
        grad_x1 = -grad_x[0]
        if self.x0_shape != self.x1_shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, self.x1_shape)

        return grad_x0, grad_x1


def sub(x0, x1):
    """Compute element-wise subtraction of two arrays.

        This function computes the element-wise subtraction of two input arrays `x0` and `x1`.
        It supports broadcasting when the input shapes are not the same.

        Args:
            x0 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The first input.
            x1 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The second input.

        Returns:
            :class:`marquetry.Container`: The result of element-wise subtracation.

        Examples:
            >>> x0 = np.array([1, 2, 3])
            >>> x1 = np.array([2, 3, 4])
            >>> sub(x0, x1)
            container([-1 -1 -1])

    """

    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    """Compute element-wise reversed order subtraction of two arrays.

        This function computes the element-wise reversed order subtraction of two input arrays `x0` and `x1`.
        It supports broadcasting when the input shapes are not the same.

        Args:
            x0 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The first input.
            x1 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The second input.

        Returns:
            :class:`marquetry.Container`: The result of element-wise reversed order subtracation.

        Examples:
            >>> x0 = np.array([1, 2, 3])
            >>> x1 = np.array([2, 3, 4])
            >>> rsub(x0, x1)
            container([1 1 1])

    """

    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Sub()(x1, x0)


class Div(Function):
    """Compute element-wise division of two arrays.

        This class computes the element-wise division of two input arrays `x0` and `x1`.
        It supports broadcasting when the input shapes are not the same.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, inputs, grad_y):
        x0, x1 = inputs
        grad_x0 = grad_y[0] / x1
        grad_x1 = grad_y[0] * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, x0.shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, x1.shape)

        return grad_x0, grad_x1


def div(x0, x1):
    """Compute element-wise division of two arrays.

        This function computes the element-wise division of two input arrays `x0` and `x1`.
        It supports broadcasting when the input shapes are not the same.

        Args:
            x0 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The first input.
            x1 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The second input.

        Returns:
            :class:`marquetry.Container`: The result of element-wise division.

        Examples:
            >>> x0 = np.array([1, 2, 3])
            >>> x1 = np.array([2, 3, 4])
            >>> div(x0, x1)
            container([0.5        0.66666667 0.75      ])

    """

    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    """Compute element-wise reversed order division of two arrays.

        This function computes the element-wise reversed order division of two input arrays `x0` and `x1`.
        It supports broadcasting when the input shapes are not the same.

        Args:
            x0 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The first input.
            x1 (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The second input.

        Returns:
            :class:`marquetry.Container`: The result of element-wise reversed order division.

        Examples:
            >>> x0 = np.array([1, 2, 3])
            >>> x1 = np.array([2, 3, 4])
            >>> rdiv(x0, x1)
            container([2.         1.5        1.33333333])

    """

    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    """Compute element-wise power of array and specified coefficient.

        This class computes the element-wise power of input array `x` and coefficient `c`.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, x, grad_y):
        c = self.c
        grad_x = c * x[0] ** (c - 1) * grad_y[0]

        return grad_x


def pow(x, c):
    """Compute element-wise power of array and specified coefficient.

        This function computes the element-wise power of input array `x` and coefficient `c`.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The input array.
            c (int): The coefficient of the power

        Returns:
            :class:`marquetry.Container`: The result of element-wise power.

        Examples:
            >>> x = np.array([1, 2, 3])
            >>> c = 4
            >>> pow(x, c)
            container([ 1 16 81])

    """

    return Pow(c)(x)

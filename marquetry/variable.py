import copy
import weakref

import numpy as np

import marquetry


# ==================================================
# VariableNode / Variable
# ==================================================
try:
    import cupy
    allow_array = (np.ndarray, cupy.ndarray)
except ImportError:
    cupy = np
    allow_array = (np.ndarray,)


class VariableNode(object):
    """Node in the backward computational graph representing arrays.

        This object represents a variable node in a computational graph.
        The node is used in backward computation to determine which gradient to be passed to each function.

        A variable node is held by the corresponding :class:`Variable` object,
        which is defined by user. :class:`Function` objects that take the variable
        as an input also hold references to the variable node.

        Note that the node does not hold a reference to the corresponding data
        array in general.
        When the function call :meth:`retain_input` and/or :meth:`retain_output`,
        node store the corresponding array.

        Users usually do not need to touch this variable node object.
        The computational graph is automatically managed by Marquetry,
        and any interface that is beneficial for users is also provided by :class:`Variable`.

        Attributes:
            creator: The creator function that generated this variable node.
            data: Data array corresponding to the variable node.
                If the data is already released, this returns `None`.
            grad: Gradient array corresponding to the variable node.
            label: Short text explaining this variable node.
            generation: Generation number of the corresponding variable, which is used for backpropagation computation.
            ndim: Number of dimensions of the data array.
            name: The corresponding variable name.

        Args:
            variable (Variable): The corresponding variable object.
            name (str): Name of the variable node.

    """

    def __init__(self, variable, name):
        self._variable = weakref.ref(variable)
        self._creator = None
        self._data = None
        self._generation = 0
        self._name = name
        self._grad = None
        self._ndim = None
        self._dtype = None
        self._shape = None

    @property
    def creator(self):
        """creator name which create this variable."""
        return self._creator

    @creator.setter
    def creator(self, func):
        self._creator = func
        if func is not None:
            self._generation = func.generation + 1

    @property
    def data(self):
        """Data array corresponding variable.

        If the data is already released, this return `None`.
        """
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
        self.set_data_type(d)

    @property
    def grad(self):
        """Gradient array corresponding variable."""
        return self._grad

    @grad.setter
    def grad(self, g):
        self._grad = g

    @property
    def label(self):
        """Short text explains this variable node."""
        if self.shape == ():
            return str(self.dtype)

        return "(%s), %s" % (", ".join(map(str, self.shape)), str(self.dtype))

    @property
    def name(self):
        """Get the name assigned to this corresponding Variable."""
        return self._name

    @property
    def generation(self):
        """Generation number of the corresponding variable, which is used for backpropagation computation."""
        return self._generation

    @property
    def ndim(self):
        """Number of dimensions of the data array."""
        return self._ndim

    @ndim.setter
    def ndim(self, data_ndim):
        self._ndim = data_ndim

    @property
    def dtype(self):
        """Dtype of the data array."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    @property
    def shape(self):
        """Shape of the data array."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    def set_creator(self, creator):
        """Set the creator function for this variable node."""
        self.creator = creator

    def unchain(self):
        """Remove the creator function reference."""
        self.creator = None

    def retain_data(self):
        """Retain variable data."""
        variable = self._variable()
        if variable is not None:
            self.data = variable.data
        else:
            raise RuntimeError("Cannot retain variable data: the variable has been already released.")

    def set_data_type(self, d):
        """Set data type and shape attributes based on the data array."""
        if d is None:
            self.dtype = None
            self.shape = None
            self.ndim = None
        else:
            self.dtype = d.dtype
            self.shape = d.shape
            self.ndim = d.ndim

    def set_grad_with_check(self, g, func, data):
        """Set the gradient while checking its type."""
        _check_grad_type(func, data, g)
        self._grad = g


def _check_grad_type(func, x, grad_x):
    """Check the type, dtype, and shape compatibility between data and gradient.

        This function is used to ensure that the type, dtype, and shape of the data and its gradient match.
        It checks if the gradient data has the same type as the input data, has the same dtype, and has the same shape.

        Args:
            func (Function or None): The function that generated the gradient, if available.
            x (Variable): The input variable.
            grad_x (Variable): The gradient variable.

        Raises:
            TypeError: If the data type or dtype of the gradient does not match the input data,
                or if the shapes do not match.

        Notes:
            If either the input data or the gradient data is `None`, no check is performed.
    """

    def make_message(message):
        if func:
            detail = "Function `{0}` ({1}) has a bug.\n".format(
                type(func).__name__, func.name
            )
            detail += '''
            Please report this error to developer with the issue trace.
            '''

        else:
            detail = ""

        detail += message
        return detail

    if x.data is None or grad_x is None:
        return

    if not isinstance(grad_x.data, type(x.data)):
        msg = ("Type of data and grad mismatch\n {} ≠ {}".format(type(x.data), type(grad_x.data)))

        raise TypeError(make_message(msg))

    if grad_x.dtype != x.data.dtype:
        raise TypeError("data and grad dtype mismatch.")

    if grad_x.shape != x.data.shape:
        raise ValueError("grad and data shape mismatch.")


class Variable(object):
    """Represents a variable with associated data and gradient in a computational graph.

        This class is used to create variables that hold data and gradients in a computational graph.
        Variables are typically　created by wrapping data arrays using this class.

        Args:
            data (:class:`numpy.ndarray` or :class:`cupy.ndarray` or None):
                The data associated with the variable.
            name (str or None): A name or label for the variable,
                which is helpful for debugging and visualization.

        Attributes:
            creator (Function or None): The function that created this variable during forward computation.
                If `None`, it means that this variable is not the result of a function.
            data (:class:`numpy.ndarray` or :class:`cupy.ndarray`): The data associated with the variable.
            grad (Variable or None): The gradient associated with this variable.
                It is usually `None` for input variables and becomes available after backward computation.
            generation (int): The generation number of the variable, used for backpropagation computation.
            name: The variable name.

        Notes:
            Variables created with this class are used in Marquetry's automatic differentiation framework.
            The `Variable` class wraps data and associates it with gradient information.
            Variables can be connected through functions to create computational graphs for automatic differentiation.

    """

    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None and not isinstance(data, allow_array):
            raise TypeError("{} is not supported.".format(type(data)))

        self._data = data
        self._name = name
        self._node = VariableNode(self, name)

        self._iteration = 0

    @property
    def creator(self):
        """The `Function` object that created this variable during forward computation.

            If `creator` is `None`, it means that this variable is not the result of a function,
            and backpropagation will not proceed beyond this variable.

            Returns:
                Function or None: The function that created this variable, or `None` if the variable is not created by a function.
        """
        return self._node.creator

    @creator.setter
    def creator(self, func):
        self._node.creator = func

    @property
    def data(self):
        """Get the data array associated with this variable.

            Returns:
                ndarray or None: The data array associated with this variable.
                    If no data is associated, it returns `None`.
        """
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
        self._node.set_data_type(d)

    @property
    def node(self):
        """Get the corresponding VariableNode for this variable.

            Returns:
                VariableNode: The VariableNode corresponding to this variable.
        """
        return self._node

    @property
    def grad(self):
        """Get the gradient array associated with this variable.

            Returns:
                Variable: The gradient variable corresponding to this variable.
        """
        return self._node.grad

    @grad.setter
    def grad(self, g):
        self._node.set_grad_with_check(g, None, self)

    @property
    def generation(self):
        """Get the generation number associated with this variable.

            Returns:
                int: The generation number used for backpropagation.
        """
        return self._node.generation

    def set_creator(self, func):
        self._node.set_creator(func)

    def unchain(self):
        """Remove the link to the creator function.

            Notes:
                Calling this method removes the computational graph connection between this variable
                and its creator function, effectively stopping backpropagation at this point.
        """
        self.creator = None

    def unchain_backward(self):
        """Unchain this variable and its ancestors in the computational graph.

            This method unchains the current variable and its ancestors by removing the links
            between them and their creator functions. It effectively stops backpropagation
            from this variable onward.

            Notes:
                This method iteratively unchains all ancestors of the current variable,
                ensuring that backpropagation is halted at this point.
        """

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            for x in f.inputs:
                if x.creator is not None:
                    add_func(x.creator)
                    x.unchain()

    def retain_data(self):
        """Retain the data of this variable within its corresponding VariableNode.

            When called, this method sets the data of this variable to the data attribute
            of its corresponding VariableNode. This is useful when you want to ensure that
            the data is retained within the computational graph for later use.

            Notes:
                Retaining data can be helpful when you need to ensure that the data remains
                accessible within the computational graph, even if the original Variable object
                goes out of scope.

        """
        self._node.data = self._data

    def clear_grad(self):
        """Clear the gradient of this variable.

            When called, this method sets the gradient of this variable to None,
            effectively clearing any gradient information associated with it.
            This can be useful when you want to remove gradient information
            from a variable(corresponding variable node).
        """

        self.grad = None

    def backward(self, retain_grad=False):
        """Perform backpropagation starting from this variable.

            During backpropagation, the `backward` method of each parent function, which created
            this variable, is called. This process calculates and stores the gradient for this
            variable (to the corresponding variable node).

            If the gradient (`grad`) of this variable is initially `None`, this method initializes
            it with an array of ones with the same shape as the data.
            This serves as the initial error for the backpropagation process.

            Args:
                retain_grad (bool): If `True`, retain the gradient computation graph
                    for potential higher-order derivatives.
                    Otherwise, release computational graph of the backpropagation.

            Notes:
                In most cases of model training, you don't need higher-order derivatives,
                so it is recommended to set `retain_grad` to `False`.
                This method go reverse through the computational graph created during forward computation
                and calculates gradients for each input variable.
                The `backward` method is typically called after the forward pass to compute gradients
                and update model parameters.

        """
        if self.creator is None:
            return

        if self.grad is None:
            xp = marquetry.cuda_backend.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            outputs = [y() for y in f.outputs]
            grad_ys = tuple(
                [None if output is None else output.grad for output in outputs])

            in_data = tuple([x.data for x in f.inputs])

            f.output_data = tuple(
                [None if y is None else y.data for y in outputs])

            grad_xs = f.backward(in_data, grad_ys)

            if not getattr(f, "_output_retain_ever", False):
                f.output_data = None

            if not isinstance(grad_xs, tuple):
                if isinstance(grad_xs, list):
                    grad_xs = tuple(grad_xs)
                else:
                    grad_xs = (grad_xs,)

            for x, grad_x in zip(f.inputs, grad_xs):
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad = x.grad + grad_x

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

            del grad_xs

    @property
    def shape(self):
        """Get the shape of the data array.

            Returns:
                tuple: A tuple representing the shape of the data array.
                    For example, for a 2D array, the shape might be (rows, columns).

        """
        return self.data.shape

    @property
    def ndim(self):
        """Get the dimensionality of the data array.

            Returns:
                int: The number of dimensions in the data array.

        """
        return self.data.ndim

    @property
    def size(self):
        """Get the total number of elements in the data array.

            Returns:
                int: The total number of elements in the data array.

        """
        return self.data.size

    @property
    def dtype(self):
        """Get the data type of the data array.

            Returns:
                numpy.dtype: The data type of the elements in the data array.

            Notes:
                The `dtype` property allows you to access the data type of the elements in the
                data array, such as float32, int64, etc.
        """
        return self.data.dtype

    @property
    def T(self):
        """Transpose the data array (reversed axis).

            Returns:
                Variable: A new variable containing the transposed data array.

            Examples:
                >>> x = Variable(np.array([[1, 2, 3], [2, 3, 4]]))
                >>> x.shape
                (2, 3)
                >>> x.T.shape
                (3, 2)
        """
        return marquetry.functions.transpose(self)

    @property
    def name(self):
        """Get the name assigned to this Variable.

            Returns:
                str or None: The name assigned to this Variable, or None if no name is assigned.

            Examples:
                >>> x = Variable(np.array(3), name="my_variable")
                >>> x.name
                'my_variable'
        """

        return self._name

    def astype(self, dtype, inplace=False):
        """Return a new variable with a specified data type.

            Args:
                dtype (str or numpy.dtype): The desired data type for the new variable.
                inplace (bool): If True, the data type of the current variable is
                    modified in-place. If False (default), a new variable with the specified
                    data type is returned.

            Returns:
                Variable: A new variable with the specified data type.

            Examples:
                >>> x = Variable(np.array([1, 2, 3], dtype='int32'))
                >>> x.dtype
                dtype('int32')
                >>> x.astype("float64", inplace=True)
                >>> x.dtype
                dtype('float64')
                >>> y = x.astype("float32", inplace=False)
                >>> x.dtype
                dtype('float64')
                >>> y.dtype
                dtype('float32')

        """

        if inplace:
            self._data = self._data.astype(dtype)
        else:
            data = self.copy()
            data = data._data.astype(dtype)
            return data

    def copy(self):
        """Create a deep copy of this variable.

            Returns:
                Variable: A new variable that is a deep copy of the current variable.

            Examples:
                >>> x = Variable(np.array([1, 2, 3]))
                >>> y = x.copy()
                >>> y.data[0] = 99
                >>> x.data[0]
                1
        """
        return copy.deepcopy(self)

    def dot(self, other):
        """Calculate the dot product between this variable and another variable or array.

            Args:
                other (Variable or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                    The other variable to compute the dot product with.

            Returns:
                Variable: A new variable containing the dot product of this variable and
                    the other variable.

            Examples:
                >>> x = Variable(np.array([1, 2, 3]))
                >>> y = Variable(np.array([4, 5, 6]))
                >>> x.dot(y).data
                matrix(32)

        """
        return marquetry.functions.matmul(self, other)

    def max(self, axis=None, keepdims=False):
        """Calculate the maximum value along the specified axis.

            Args:
                axis (int or tuple or None): The axis or axes along which to compute the maximum value.
                    If None (default), the maximum value is calculated over all elements.
                keepdims (bool): If True, the dimensions of the output are retained,
                    and the result will have the same number of dimensions as the original variable.
                    If False (default), dimensions of size 1 are removed from the output.

            Returns:
                Variable: A new variable containing the maximum value along the specified axis or axes.

            Examples:
                >>> x = Variable(np.array([[1, 2], [3, 4]]))
                >>> x.max()
                matrix(4)
                >>> x.max(axis=0)
                matrix([3 4])
                >>> x.max(axis=1, keepdims=True)
                matrix([[2]
                        [4]])
        """
        return marquetry.functions.max(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        """Calculate the mean (average) along the specified axis.

            Args:
                axis (int or tuple or None): The axis or axes along which to compute the mean.
                    If None (default), the mean is calculated over all elements.
                keepdims (bool): If True, the dimensions of the output are retained,
                    and the result will have the same number of dimensions as the original variable.
                    If False (default), dimensions of size 1 are removed from the output.

            Returns:
                Variable: A new variable containing the mean along the specified axis or axes.

            Examples:
                >>> x = Variable(np.array([[1, 2], [3, 4]]))
                >>> x.mean()
                matrix(2.5)
                >>> x.mean(axis=0)
                matrix([2. 3.])
                >>> x.mean(axis=1, keepdims=True)
                matrix([[1.5]
                        [3.5]])
        """

        return marquetry.functions.mean(self, axis, keepdims)

    def repeat(self, repeats, axis=None):
        """Repeat elements of the variable along the specified axis.

            Args:
                repeats (int or list of ints or tuple of ints):
                    The number of times to repeat each element along the specified axis.
                    If it is an int, all elements are repeated that number of times.
                    If it is a sequence of ints, it must have the same length as the specified axis.
                axis (int): The axis along which to repeat the elements.
                    If None (default), the variable is flattened before repeating.

            Returns:
                Variable: A new variable containing the repeated elements along the specified
                    axis.

            Examples:
                >>> x = Variable(np.array([1, 2, 3]))
                >>> x.repeat(3)
                array([1, 1, 1, 2, 2, 2, 3, 3, 3])
                >>> x.repeat([2, 3, 1])
                array([1, 1, 2, 2, 2, 3])
        """
        return marquetry.functions.repeat(self, repeats, axis)

    def reshape(self, *shape):
        """Reshape the variable to the specified shape.

            Args:
                *shape: One or more integers or a tuple of integers specifying the desired
                    shape of the reshaped variable.

            Returns:
                Variable: A new variable with the specified shape.

            Examples:
                >>> x = Variable(np.array([1, 2, 3, 4, 5, 6]))
                >>> x.reshape(2, 3)
                matrix([[1 2 3]
                        [4 5 6]])
                >>> x.reshape(-1, 2)
                matrix([[1 2]
                        [3 4]
                        [5 6]])

        """

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return marquetry.functions.reshape(self, shape)

    def sum(self, axis=None, keepdims=False):
        """Calculate the sum of the variable's elements along the specified axis.

            Args:
                axis (int or tuple or None): The axis or axes along which to compute the sum.
                    If None (default), the sum is calculated over all elements.
                keepdims (bool): If True, the dimensions of the output are retained,
                    and the result will have the same number of dimensions as the original variable.
                    If False (default), dimensions of size 1 are removed from the output.

            Returns:
                Variable: A new variable containing the sum along the specified axis or axes.

            Examples:
                >>> x = Variable(np.array([[1, 2], [3, 4]]))
                >>> x.sum()
                matrix(10)
                >>> x.sum(axis=0)
                matrix([4 6])
                >>> x.sum(axis=1, keepdims=True)
                matrix([[3]
                        [7]])
        """

        return marquetry.functions.sum(self, axis, keepdims)

    def squeeze(self, axis=None):
        """Remove single-dimensional entries from the variable's shape.

            Args:
                axis (int or tuple of ints or list of ints or None): The axis along which to remove single-dimensional entries.
                    If None (default), all single-dimensional entries are removed.

            Returns:
                Variable: A new variable with the specified single-dimensional entries removed.

            Examples:
                >>> x = Variable(np.array([[[1]], [[2]], [[3]]]))
                >>> x.shape
                (3, 1, 1)
                >>> y = x.squeeze()
                >>> y.shape
                (3,)
                >>> z = x.squeeze(axis=1)
                >>> z.shape
                (3, 1)
        """
        return marquetry.functions.squeeze(self, axis)

    def to_numpy(self):
        """Convert the variable to a NumPy array.

            Returns:
                np.ndarray: A NumPy array representing the data of the variable.

            Raises:
                TypeError: If the variable has a gradient matrix, it cannot be converted to a NumPy array.

            Examples:
                >>> x = Variable(np.array([1, 2, 3]))
                >>> y = x.to_numpy()
                >>> type(y)
                <class 'numpy.ndarray'>
        """

        if self.grad is not None:
            raise TypeError("The matrix has gradient so it can't convert to numpy array. please do clear_grad first.")
        return self.data

    def transpose(self, *axes):
        """Transpose the variable's data array.

            Args:
                *axes: One or more integers or a tuple of integers specifying the desired
                    order of dimensions in the transposed variable. If no axes are provided,
                    the variable is fully transposed.

            Returns:
                Variable: A new variable with the dimensions reordered as specified.

            Examples:
                >>> x = Variable(np.array([[[1], [2], [3]], [[4], [5], [6]]]))
                >>> x.shape
                (2, 3, 1)
                >>> y = x.transpose()
                >>> y.shape
                (1, 3, 2)
                >>> z = x.transpose(1, 0, 2)
                >>> z.shape
                (3, 2, 1)

        """

        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]

        return marquetry.functions.transpose(self, axes)

    def unsqueeze(self, axis):
        """Add a single-dimensional entry to the variable's shape.

            Args:
                axis (int): The axis along which to add a single-dimensional entry.

            Returns:
                Variable: A new variable with a single-dimensional entry added along the specified axis.

            Examples:
                >>> x = Variable(np.array([1, 2, 3]))
                >>> y = x.unsqueeze(0)
                >>> y.shape
                (1, 3)
                >>> z = x.unsqueeze(1)
                >>> z.shape
                (3, 1)

        """

        return marquetry.functions.unsqueeze(self, axis)

    def to_cpu(self):
        """Move the variable's data array to the CPU memory(numpy.ndarray).

            This method converts the variable's data array from GPU memory to CPU memory.
            If the variable's data is already in CPU memory or if it is `None`, no action is taken.

        """

        if self.data is None:
            return

        self._data = marquetry.cuda_backend.as_numpy(self.data)

        node = self._node
        if node.data is not None:
            node.retain_data()

    def to_gpu(self):
        """Move the variable's data array to GPU memory(cupy.ndarray).

            This method converts the variable's data array from CPU memory to GPU memory.
            If the variable's data is already in GPU memory or if it is `None`, no action is taken.

        """

        if self.data is not None:
            self._data = marquetry.cuda_backend.as_cupy(self.data)

            node = self._node
            if node.data is not None:
                node.retain_data()

    def __matmul__(self, other):
        return marquetry.functions.matmul(self, other)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "matrix(None)"
        p = str(self.data).replace("\n", "\n" + " " * 7)
        return "matrix(" + p + ")"

    def __add__(self, other):
        return marquetry.functions.add(self, other)

    def __radd__(self, other):
        return marquetry.functions.add(self, other)

    def __iadd__(self, other):
        return marquetry.functions.add(self, other)

    def __sub__(self, other):
        return marquetry.functions.sub(self, other)

    def __rsub__(self, other):
        return marquetry.functions.rsub(self, other)

    def __isub__(self, other):
        return marquetry.functions.sub(self, other)

    def __mul__(self, other):
        return marquetry.functions.mul(self, other)

    def __rmul__(self, other):
        return marquetry.functions.mul(self, other)

    def __imul__(self, other):
        return marquetry.functions.mul(self, other)

    def __neg__(self):
        return marquetry.functions.neg(self)

    def __truediv__(self, other):
        return marquetry.functions.div(self, other)

    def __rtruediv__(self, other):
        return marquetry.functions.rdiv(self, other)

    def __itruediv__(self, other):
        return marquetry.functions.div(self, other)

    def __pow__(self, power):
        return marquetry.functions.pow(self, power)

    def __getitem__(self, item):
        return marquetry.functions.get_item(self, item)

    def __eq__(self, other):
        other = as_variable(as_array(other, marquetry.cuda_backend.get_array_module(self.data)))
        return self.data == other.data

    def __ne__(self, other):
        other = as_variable(as_array(other, marquetry.cuda_backend.get_array_module(self.data)))
        return self.data != other.data

    def __lt__(self, other):
        other = as_variable(as_array(other, marquetry.cuda_backend.get_array_module(self.data)))
        return self.data < other.data

    def __gt__(self, other):
        other = as_variable(as_array(other, marquetry.cuda_backend.get_array_module(self.data)))
        return self.data > other.data

    def __le__(self, other):
        other = as_variable(as_array(other, marquetry.cuda_backend.get_array_module(self.data)))
        return self.data <= other.data

    def __ge__(self, other):
        other = as_variable(as_array(other, marquetry.cuda_backend.get_array_module(self.data)))
        return self.data >= other.data

    def __bool__(self):
        return self.data.__bool__()

    def __hash__(self):
        return super(Variable, self).__hash__()


class Parameter(Variable):
    """A special type of variable used to represent model parameters.

        In deep learning frameworks, the `Parameter` class is typically used to
        represent parameters like weights and biases.
        It inherits from the `Variable` class, which means it can be used like a regular variable
        but often carries specific attributes or methods tailored for model parameters.

        This class provides the flexibility to add parameter-specific functionality
        or attributes when needed.

        Examples:
            >>> weight = Parameter(np.random.randn(10, 5), name="weight")
            >>> bias = Parameter(np.zeros(5), name="bias")
    """
    pass


def as_variable(obj):
    """Convert an object to a Variable if it is not already one.

        Args:
            obj: The object to be converted to a Variable.

        Returns:
            Variable: The Variable representation of the input object if it is not already a Variable.

        Examples:
            >>> x = np.array(5)
            >>> type(x)
            <class 'numpy.ndarray'>
            >>> y = as_variable(x)
            >>> type(y)
            <class 'marquetry.variable.Variable'>
    """

    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x, array_type=np):
    """Convert an object to a ndarray whose specified module.

        If the input object is a scalar, it is converted to a ndarray whose specified module.
        Otherwise, the object is returned as is.

        Args:
            x(int or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The object to be converted to a NumPy array if it is a scalar.
            array_type (numpy or cupy): The module managed the actual array object
                to use when converting a scalar to a ndarray, selectable cupy or numpy.
                Defaults to `np`, which is the NumPy module.

        Returns:
            :class:`np.ndarray` or :class:`cp.ndarray`:
                The ndarray representation of the input object if it is a scalar.
                Otherwise, the input object itself.

        Examples:
            >>> x = 5
            >>> y = as_array(x)
            >>> type(y)
            <class 'numpy.ndarray'>

    """

    if np.isscalar(x):
        return array_type.array(x)
    return x


def array(x, name=None):
    """Create a Variable from a given ndarray whose specified module(CuPy or NumPy).

        Args:
            x(:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The ndarray to be wrapped in a Variable.
            name (str or None): The name to assign to the Variable.

        Returns:
            Variable: A Variable object that wraps the input NumPy array.

        Examples:
            >>> import numpy as np
            >>> arr = np.array([1, 2, 3])
            >>> var = array(arr, name="my_variable")
            >>> type(var)
            <class 'marquetry.variable.Variable'>
            >>> var.name
            'my_variable'
    """

    return Variable(x, name=name)

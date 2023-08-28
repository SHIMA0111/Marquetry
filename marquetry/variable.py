import copy
import weakref

import numpy as np

import marquetry


# ==================================================
# Variable / Function 2
# ==================================================
try:
    import cupy
    allow_array = (np.ndarray, cupy.ndarray)
except ImportError:
    cupy = np
    allow_array = (np.ndarray,)


class VariableNode(object):
    def __init__(self, variable, name):
        self._variable = weakref.ref(variable)
        self._creator = None
        self._data = None
        self._generation = 0
        self.name = name
        self._grad = None
        self._ndim = None

    @property
    def creator(self):
        return self._creator

    @creator.setter
    def creator(self, func):
        self._creator = func
        if func is not None:
            self._generation = func.generation + 1

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
        self._set_data_type(d)

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, g):
        self._grad = g

    @property
    def label(self):
        if self.shape == ():
            return str(self.dtype)

        return "(%s), %s" % (", ".join(map(str, self.shape)), str(self.dtype))

    @property
    def generation(self):
        return self._generation

    @property
    def ndim(self):
        return self._ndim

    @ndim.setter
    def ndim(self, data_ndim):
        self._ndim = data_ndim

    def set_creator(self, creator):
        self.creator = creator

    def unchain(self):
        self.creator = None

    def retain_data(self):
        variable = self._variable()
        if variable is not None:
            self.data = variable.data
        else:
            raise RuntimeError("Cannot retain variable data: the variable has been already released.")

    def _set_data_type(self, d):
        if d is None:
            self.dtype = None
            self.shape = None
            self.ndim = None
        else:
            self.dtype = d.dtype
            self.shape = d.shape
            self.ndim = d.ndim

    def set_grad_with_check(self, g, func, data):
        _check_grad_type(func, data, g)
        self._grad = g


def _check_grad_type(func, x, grad_x):
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
        return self._node.creator

    @creator.setter
    def creator(self, func):
        self._node.creator = func

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
        self._node._set_data_type(d)

    @property
    def node(self):
        return self._node

    @property
    def grad(self):
        return self._node.grad

    @grad.setter
    def grad(self, g):
        self._node.set_grad_with_check(g, None, self)

    @property
    def generation(self):
        return self._node.generation

    def set_creator(self, func):
        self._node.set_creator(func)

    def unchain(self):
        self.creator = None

    def unchain_backward(self):
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
        self._node.data = self._data

    def clear_grad(self):
        self.grad = None

    def backward(self):
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

            for y in f.outputs:
                y().grad = None

            del grad_xs

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return marquetry.functions.transpose(self)

    def copy(self):
        return copy.deepcopy(self)

    def dot(self, other):
        return marquetry.functions.matmul(self, other)

    def max(self, axis=None, keepdims=False):
        return marquetry.functions.max(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return marquetry.functions.mean(self, axis, keepdims)

    def repeat(self, repeats, axis=None):
        return marquetry.functions.repeat(self, repeats, axis)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return marquetry.functions.reshape(self, shape)

    def sum(self, axis=None, keepdims=False):
        return marquetry.functions.sum(self, axis, keepdims)

    def squeeze(self, axis):
        return marquetry.functions.squeeze(self, axis)

    def to_numpy(self):
        if self.grad is not None:
            raise TypeError("Having gradient matrix can't convert to numpy array.")
        return self.data

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]

        return marquetry.functions.transpose(self, axes)

    def unsqueeze(self, axis):
        return marquetry.functions.unsqueeze(self, axis)

    def to_cpu(self):
        if self.data is None:
            return

        self._data = marquetry.cuda_backend.as_numpy(self.data)

        node = self._node
        if node._data is not None:
            node.retain_data()

    def to_gpu(self):
        if self.data is not None:
            self._data = marquetry.cuda_backend.as_cupy(self.data)

            node = self._node
            if node._data is not None:
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
    pass


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x, array_type=np):
    if np.isscalar(x):
        return array_type.array(x)
    return x


def array(x):
    return Variable(x)

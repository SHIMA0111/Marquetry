import contextlib
import copy
import weakref
import os

import numpy as np

import marquetry


# ==================================================
# Variable / Function
# ==================================================
try:
    import cupy
    allow_array = (np.ndarray, cupy.ndarray)
except ImportError:
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

        self.generation = 0

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
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __iadd__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return rsub(self, other)

    def __isub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __imul__(self, other):
        return mul(self, other)

    def __neg__(self):
        return neg(self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return rdiv(self, other)

    def __itruediv__(self, other):
        return div(self, other)

    def __pow__(self, power):
        return pow(self, power)

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


class Function(object):
    generation = 0
    _input_indexes_to_retain = None
    _output_indexes_to_retain = None
    _output_retain_ever = None

    inputs = None
    outputs = None
    output_data = None

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]

        xp = marquetry.cuda_backend.get_array_module(xs[0])

        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y, xp)) for y in ys]

        if marquetry.Config.train_mode:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = tuple([x.node for x in inputs])
            self.outputs = tuple([weakref.ref(output.node) for output in outputs])

            input_indexes_to_retain = self._input_indexes_to_retain
            if input_indexes_to_retain is None:
                input_indexes_to_retain = range(len(inputs))
            for index in input_indexes_to_retain:
                inputs[index].retain_data()

            self._input_indexes_to_retain = None

            output_indexes_to_retain = self._output_indexes_to_retain
            if output_indexes_to_retain is not None:
                for index in output_indexes_to_retain:
                    outputs[index].retain_data()

            self._output_indexes_to_retain = None

        return outputs if len(outputs) > 1 else outputs[0]

    @property
    def name(self):
        return self.__class__.__name__

    def unchain(self):
        for y in self.outputs:
            y_ref = y()
            if y_ref is not None:
                y_ref.unchain()

        self.inputs = None

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, *grad_ys):
        raise NotImplementedError()

    def retain_inputs(self, indexes):
        self._input_indexes_to_retain = indexes

    def retain_outputs(self, indexes, retain_ever=False):
        self._output_indexes_to_retain = indexes
        if retain_ever:
            self._output_retain_ever = retain_ever


# ==================================================
# Basic formula / operator overload
# ==================================================
class Add(Function):
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
    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
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
    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        self.retain_inputs(())
        return -x

    def backward(self, x, grad_y):
        return -grad_y[0]


def neg(x):
    return Neg()(x)


class Sub(Function):
    def __init__(self):
        self.x0_shape = None
        self.x1_shape = None

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1

        self.retain_inputs(())
        return y

    def backward(self, grad_x):
        grad_x0 = grad_x[0]
        grad_x1 = -grad_x[0]
        if self.x0_shape != self.x1_shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, self.x1_shape)

        return grad_x0, grad_x1


def sub(x0, x1):
    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Sub()(x1, x0)


class Div(Function):
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
    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, marquetry.cuda_backend.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
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
    return Pow(c)(x)


# ==================================================
# Config
# ==================================================
class Config(object):
    train_mode = True
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".marquetry")


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def test_mode():
    return using_config('train_mode', False)

import contextlib
import copy
import weakref
import os

import numpy as np

import marquetry


# ==================================================
# Variable / Function
# ==================================================
class Variable(object):
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, (np.ndarray, list)):
                raise TypeError("{} is not supported.".format(type(data)))

        self.data = np.array(data) if data is not None else data
        self.name = name

        self.generation = 0

        self.grad = None
        self.creator = None

        self._iteration = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

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

    def clear_grad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

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
            grad_ys = [output().grad for output in f.outputs]

            grad_xs = f.backward(*grad_ys)
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
        other = as_variable(as_array(other))
        return self.data == other.data

    def __ne__(self, other):
        other = as_variable(as_array(other))
        return self.data != other.data

    def __lt__(self, other):
        other = as_variable(as_array(other))
        return self.data < other.data

    def __gt__(self, other):
        other = as_variable(as_array(other))
        return self.data > other.data

    def __le__(self, other):
        other = as_variable(as_array(other))
        return self.data <= other.data

    def __ge__(self, other):
        other = as_variable(as_array(other))
        return self.data >= other.data

    def __bool__(self):
        return self.data.__bool__()


class Parameter(Variable):
    pass


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def array(x):
    return Variable(x)


class Function(object):
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, *grad_ys):
        raise NotImplementedError()


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

        return y

    def backward(self, grad_y):
        grad_x0, grad_x1 = grad_y, grad_y
        if self.x0_shape != self.x1_shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, self.x1_shape)

        return grad_x0, grad_x1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, grad_y):
        x0, x1 = self.inputs
        grad_x0 = grad_y * x1
        grad_x1 = grad_y * x0
        if x0.shape != x1.shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, x0.shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, x1.shape)

        return grad_x0, grad_x1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, grad_y):
        return -grad_y


def neg(x):
    return Neg()(x)


class Sub(Function):
    def __init__(self):
        self.x0_shape = None
        self.x1_shape = None

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1

        return y

    def backward(self, grad_x):
        grad_x0 = grad_x
        grad_x1 = -grad_x
        if self.x0_shape != self.x1_shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, self.x1_shape)

        return grad_x0, grad_x1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, grad_y):
        x0, x1 = self.inputs
        grad_x0 = grad_y / x1
        grad_x1 = grad_y * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:
            grad_x0 = marquetry.functions.sum_to(grad_x0, x0.shape)
            grad_x1 = marquetry.functions.sum_to(grad_x1, x1.shape)

        return grad_x0, grad_x1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, grad_y):
        x,  = self.inputs
        c = self.c
        grad_x = c * x ** (c - 1) * grad_y

        return grad_x


def pow(x, c):
    return Pow(c)(x)


# ==================================================
# Config
# ==================================================
class Config(object):
    train_mode = True
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ml_modules")


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

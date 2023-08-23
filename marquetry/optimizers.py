import math

import numpy as np

from marquetry import cuda_backend


# ===========================================================================
# Optimizer base class
# ===========================================================================
class Optimizer(object):
    def __init__(self):
        self.target = None
        self.additional_hooks = []

    def prepare(self, target):
        self.target = target

        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.additional_hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, hook):
        self.additional_hooks.append(hook)


# ===========================================================================
# Hooks
# ===========================================================================
class WeightDecay(object):
    def __init__(self, decay, method="l2"):
        self.decay = decay
        self.method = method

    def __call__(self, params):
        for param in params:
            if self.method == "l1":
                param.grad.data += self.decay * abs(param.data)
            else:
                param.grad.data += self.decay * param.data ** 2


class ClipGrad(object):
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, params):
        total_norm = 0
        for param in params:
            total_norm += (param.grad.data ** 2).sum()

        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-15)

        if rate < 1:
            for param in params:
                param.grad.data *= rate


# =============================================================================
# Optimizer
# =============================================================================
class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.lr = learning_rate

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, learning_rate=0.01, decay=0.9):
        super().__init__()
        self.lr = learning_rate
        self.momentum = decay

        self.momentum_vector = {}

    def update_one(self, param):
        v_key = id(param)

        if v_key not in self.momentum_vector:
            xp = cuda_backend.get_array_module(param.data)
            self.momentum_vector[v_key] = xp.zeros_like(param.data)

        pre_vector = self.momentum_vector[v_key]
        pre_vector *= self.momentum
        pre_vector -= (1 - self.momentum) * param.grad.data

        param.data += pre_vector


class AdaGrad(Optimizer):
    def __init__(self, learning_rate=0.001, eps=1e-8):
        super().__init__()
        self.lr = learning_rate
        self.eps = eps

        self.histories = {}

    def update_one(self, param):
        h_key = id(param)

        xp = cuda_backend.get_array_module(param.data)
        if h_key not in self.histories:
            self.histories[h_key] = xp.zeros_like(param.data)

        history = self.histories[h_key]
        grad = param.grad.data

        history += grad ** 2
        param.data -= self.lr * grad / (xp.sqrt(history) + self.eps)


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.01, decay=0.99, eps=1e-8):
        super().__init__()
        self.lr = learning_rate
        self.decay = decay
        self.eps = eps

        self.histories = {}

    def update_one(self, param):
        h_key = id(param)

        xp = cuda_backend.get_array_module(param.data)
        if h_key not in self.histories:
            self.histories[h_key] = xp.zeros_like(param.data)

        history = self.histories[h_key]
        grad = param.grad.data

        history *= self.decay
        history += (1 - self.decay) * grad ** 2

        param.data -= self.lr * grad / (xp.sqrt(history) + self.eps)


class Adam(Optimizer):
    def __init__(self, base_learning_rate=0.001, first_decay=0.9, second_decay=0.999, eps=1e-8):
        super().__init__()
        self.blr = base_learning_rate
        self.fd = first_decay
        self.sd = second_decay
        self.eps = eps

        self.iters = 0

        self.momentum_vector = {}
        self.histories = {}

    def update(self):
        self.iters += 1
        super().update()

    def update_one(self, param):
        param_key = id(param)

        xp = cuda_backend.get_array_module(param.data)
        if param_key not in self.momentum_vector:
            self.momentum_vector[param_key] = xp.zeros_like(param.data)
            self.histories[param_key] = xp.zeros_like(param.data)

        vector, history = self.momentum_vector[param_key], self.histories[param_key]

        grad = param.grad.data

        vector *= self.fd
        vector += (1 - self.fd) * grad

        history *= self.sd
        history += (1 - self.sd) * grad ** 2

        param.data -= self.lr * vector / (xp.sqrt(history) + self.eps)

    @property
    def lr(self):
        correction1 = 1. - math.pow(self.fd, self.iters)
        correction2 = 1. - math.pow(self.sd, self.iters)

        return self.blr * math.sqrt(correction2) / (correction1 + self.eps)

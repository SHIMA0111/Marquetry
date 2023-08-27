import math


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

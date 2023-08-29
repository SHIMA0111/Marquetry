import math

from marquetry import cuda_backend
from marquetry.optimizer import Optimizer


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

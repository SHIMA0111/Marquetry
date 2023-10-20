import math

from marquetry import cuda_backend
from marquetry.optimizer import Optimizer


class Adam(Optimizer):
    """Adaptive Moment Estimation(Adam) optimizer for updating model parameters during training.

        Adam is an optimization algorithm that combines the concepts of MomentumSGD and RMSProp to
        adaptively adjust the learning rates for each parameter.
        It is known for its effectiveness in a wide range of optimization problems.

        One of the peculiarities point of Adam optimizer is the adjustable learning rate based on the current iteration.
        It combines the effects of both first and second moment estimates.


        Args:
            lr (float): The base learning rate for updating parameters.
                Default is 0.001.
            first_decay (float): The decay rate for the first moment estimate (momentum).
                Default is 0.9.
            second_decay (float): The decay rate for the second moment estimate (RMSProp).
                Default is 0.999.
            eps (float): A small value (epsilon) added to the denominator to prevent division by zero.
                Default is 1e-8.

        Tip:
            When you would like to optimize your model's parameter,
            please set the model to this using ``prepare`` method.

        Examples:
            >>> optimizer = Adam()
            >>> model = marquetry.models.MLP([128, 256, 64, 10])
            >>> optimizer.prepare(model)
            >>> optimizer.update()

    """

    def __init__(self, lr=0.001, first_decay=0.9, second_decay=0.999, eps=1e-8):
        super().__init__()
        self.blr = lr
        self.fd = first_decay
        self.sd = second_decay
        self.eps = eps

        self.iters = 0

        self.m = {}
        self.v = {}

    def update(self):
        self.iters += 1
        super().update()

    def _update_one(self, param):
        param_key = id(param)

        xp = cuda_backend.get_array_module(param.data)
        if param_key not in self.m:
            self.m[param_key] = xp.zeros_like(param.data)
            self.v[param_key] = xp.zeros_like(param.data)

        m, v = self.m[param_key], self.v[param_key]

        grad = param.grad.data

        m *= self.fd
        m += (1 - self.fd) * grad

        v *= self.sd
        v += (1 - self.sd) * grad ** 2

        param.data -= self.lr * m / (xp.sqrt(v) + self.eps)

    @property
    def lr(self):
        correction1 = 1. - math.pow(self.fd, self.iters)
        correction2 = 1. - math.pow(self.sd, self.iters)

        return self.blr * math.sqrt(correction2) / (correction1 + self.eps)

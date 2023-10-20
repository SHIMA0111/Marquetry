import math

from marquetry import cuda_backend
from marquetry.optimizer import Optimizer


class AdaMax(Optimizer):
    """AdaMax optimizer for updating model parameters during training.

        AdaMax is an optimization algorithm that extends the ideas of Adam by using the L-infinity norm of the
        gradient rather than the L-2 norm for weight updates.
        It provides a robust alternative to Adam with a stable learning rate.

        Args:
            lr (float): The learning rate for updating parameters. Default is 0.002.
            first_decay (float): The first moment decay factor for controlling the influence of past gradients.
                Default is 0.9.
            second_decay (float): The second moment decay factor for controlling the influence of
                past absolute gradients. Default is 0.999.
            eps (float): A small constant added to prevent division by zero. Default is 1e-7.

        Tip:
            When you would like to optimize your model's parameter,
            please set the model to this using ``prepare`` method.

        Examples:
            >>> optimizer = AdaMax()
            >>> model = marquetry.models.MLP([128, 256, 64, 10])
            >>> optimizer.prepare(model)
            >>> optimizer.update()

    """

    def __init__(self, lr=0.002, first_decay=0.9, second_decay=0.999, eps=1e-7):
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
        key = id(param)

        xp = cuda_backend.get_array_module(param.data)

        if key not in self.m:
            self.m[key] = xp.zeros_like(param.data)
            self.v[key] = xp.zeros_like(param.data)

        m, v = self.m[key], self.v[key]
        fd = self.fd
        sd = self.sd
        grad = param.grad.data
        eps = self.eps

        m *= fd
        m += (1. - fd) * grad

        v[...] = xp.maximum(sd * v, xp.abs(grad))
        param.data -= self.lr * m / (v + eps)

    @property
    def lr(self):
        adoptive_param = 1. - math.pow(self.fd, self.iters)

        return self.blr / (adoptive_param + self.eps)

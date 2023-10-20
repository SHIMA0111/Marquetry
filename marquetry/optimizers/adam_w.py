import math

from marquetry import cuda_backend
from marquetry.optimizer import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer for updating model parameters during training.

        AdamW is an optimization algorithm that combines the ideas of Adam (adaptive moment estimation) with
        weight decay regularization (hence the 'W' in the name).
        Weight decay helps prevent overfitting by penalizing large weights in the model.

        Args:
            lr (float): The learning rate for updating parameters.
                Default is 0.001.
            first_decay (float): The first moment decay factor, controlling the influence of past gradients.
                Default is 0.9.
            second_decay (float): The second moment decay factor, controlling the influence of past squared gradients.
                Default is 0.999.
            weight_decay (float): The weight decay factor to apply regularization. Default is 0.004.
            eps (float): A small constant added to prevent division by zero. Default is 1e-8.

        Tip:
            When you would like to optimize your model's parameter,
            please set the model to this using ``prepare`` method.

        Examples:
            >>> optimizer = AdamW()
            >>> model = marquetry.models.MLP([128, 256, 64, 10])
            >>> optimizer.prepare(model)
            >>> optimizer.update()

    """

    def __init__(self, lr=0.001, first_decay=0.9, second_decay=0.999, weight_decay=0.004, eps=1e-8):
        super().__init__()

        self.blr = lr
        self.fd = first_decay
        self.sd = second_decay
        self.wd = weight_decay
        self.eps = eps

        self.m = {}
        self.v = {}
        self.iters = 0

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

        m *= fd
        m += (1 - fd) * grad

        v *= sd
        v += (1 - sd) * grad ** 2

        param.data -= self.lr * m / (xp.sqrt(v) + self.eps) + self.wd * self.blr * param.data

    @property
    def lr(self):
        correction1 = 1 - math.pow(self.fd, self.iters)
        correction2 = 1 - math.pow(self.sd, self.iters)

        return self.blr * math.sqrt(correction2) / (correction1 + self.eps)

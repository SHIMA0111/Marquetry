import math

from marquetry import cuda_backend
from marquetry.optimizer import Optimizer


class Nadam(Optimizer):
    """Nesterov Adaptive Moment Estimation (Nadam) optimizer for updating model parameters during training.

        Nadam is an optimization algorithm that uses a combination of adaptive moment estimation (Adam) and Nesterov
        accelerated gradient (NAG) methods to update model parameters.
        It adapts the learning rates for each parameter to accelerate convergence and avoid locally optimal solutions.

        This algorithm was inspired by momentum in physics.

        Args:
            lr (float): The learning rate for updating parameters.
                Default is 0.001.
            first_decay (float): The decay factor for the first moment (momentum).
                Default is 0.9.
            second_decay (float): The decay factor for the second moment (moving average of squared gradients).
                Default is 0.999.
            time_decay (float): The time decay factor for the first_decay with time consideration.
                Default is 0.95.
            eps (float): A small constant to prevent division by zero.
                Default is 1e-7.


        Tip:
            When you would like to optimize your model's parameter,
            please set the model to this using ``prepare`` method.

        Examples:
            >>> optimizer = Nadam()
            >>> model = marquetry.models.MLP([128, 256, 64, 10])
            >>> optimizer.prepare(model)
            >>> optimizer.update()

    """

    def __init__(self, lr=0.001, first_decay=0.9, second_decay=0.999, time_decay=0.95, eps=1e-7):
        super().__init__()

        self.lr = lr
        self.fd = first_decay
        self.sd = second_decay
        self.decay = time_decay
        self.eps = eps

        self.iters = 0
        self.fd_product_count = 1
        self.fd_product = 1.
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

        current_step = self.iters
        next_step = self.iters + 1
        decay = self.decay

        m, v = self.m[key], self.v[key]
        fd = self.fd
        sd = self.sd
        grad = param.grad.data

        current_fd = fd * (1.0 - math.pow(decay, current_step))
        next_fd = fd * (1.0 - math.pow(decay, next_step))

        def get_fd_product():
            # prevent the duplicate computation when call this func in the same iteration.
            if self.fd_product_count == self.iters + 2:
                return self.fd_product
            else:
                current_fd_product = self.fd_product * current_fd
                self.fd_product = current_fd_product
                self.fd_product_count += 1

                return current_fd_product

        fd_product = get_fd_product()
        fd_product_next = fd_product * next_fd

        m *= fd
        m += (1 - fd) * grad

        v *= sd
        v += (1 - sd) * grad ** 2

        grad_hat = grad / (1 - fd_product)
        m_hat = m / (1 - fd_product_next)
        v_hat = v / (1 - math.pow(sd, self.iters))
        m_bar = (1 - fd_product) * grad_hat + fd_product_next * m_hat

        param.data -= self.lr * m_bar / (xp.sqrt(v_hat) + self.eps)

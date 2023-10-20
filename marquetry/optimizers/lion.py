from marquetry import cuda_backend
from marquetry.optimizer import Optimizer


class Lion(Optimizer):
    """EvoLved Sign Momentum (LION) optimizer for updating model parameters during training.

        LION is an optimization algorithm that is explored by google AutoML to reduce the memory efficient and
        convergence time than famous traditional optimizers like Adam.

        The one of the speciality of LION is that LION using only sign of the momentum param.
        The size of the updating only depends on the learning rate.

        This algorithm is very simple and track only momentum so the convergence time is very speedy and
        the memory efficient is also high.

        Args:
            lr (float): The learning rate for updating parameters.
                Default is 0.0001.
            first_decay (float): The first moment decay factor,
                controlling the influence of past gradients for updating.
                Default is 0.9.
            second_decay (float): The second moment decay factor,
                controlling the influence of past gradients for stored.
                Default is 0.99.
            weight_decay (float): The weight decay factor, applying regularization to the model parameters.
                Default is 0.0.

        Tip:
            When you would like to optimize your model's parameter,
            please set the model to this using ``prepare`` method.

        Examples:
            >>> optimizer = Lion()
            >>> model = marquetry.models.MLP([128, 256, 64, 10])
            >>> optimizer.prepare(model)
            >>> optimizer.update()

    """

    def __init__(self, lr=0.0001, first_decay=0.9, second_decay=0.99, weight_decay=0.0):
        super().__init__()

        self.lr = lr
        self.fd = first_decay
        self.sd = second_decay
        self.wd = weight_decay

        self.m = {}

    def _update_one(self, param):
        key = id(param)

        xp = cuda_backend.get_array_module(param.data)

        if key not in self.m:
            self.m[key] = xp.zeros_like(param.data)

        m = self.m[key]
        fd = self.fd
        sd = self.sd
        grad = param.grad.data

        c = fd * m + (1 - fd) * grad

        param.data -= self.lr * (xp.sign(c) + self.wd * param.data)

        m *= sd
        m += (1 - sd) * grad

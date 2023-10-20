from marquetry import cuda_backend
from marquetry.optimizer import Optimizer


class AdaDelta(Optimizer):
    """AdaDelta optimizer for updating model parameters during training.

        AdaDelta is an optimization algorithm that adapts the learning rate based on the moving average
        of squared gradients.

        AdaDelta is the extends the idea of AdaGrad( :class:`marquetry.optimizers.AdaGrad` ).

        It helps improve training stability and convergence by eliminating the need for manually setting a learning rate.

    Args:
        lr (float): The learning rate for updating parameters.
            Default is 1.0.
        rho (float): The decay factor for the moving averages of squared gradients.
            Default is 0.95.
        eps (float): A small constant added to prevent division by zero.
            Default is 1e-6.

    Tip:
        When you would like to optimize your model's parameter,
        please set the model to this using ``prepare`` method.

    Examples:
        >>> optimizer = AdaDelta()
        >>> model = marquetry.models.MLP([128, 256, 64, 10])
        >>> optimizer.prepare(model)
        >>> optimizer.update()



    """

    def __init__(self, lr=1.0, rho=0.95, eps=1e-6):
        super().__init__()

        self.lr = lr
        self.rho = rho
        self.eps = eps

        self.h = {}
        self.s = {}

    def _update_one(self, param):
        key = id(param)

        xp = cuda_backend.get_array_module(param.data)
        if key not in self.h:
            self.h[key] = xp.zeros_like(param.data)
            self.s[key] = xp.zeros_like(param.data)

        h = self.h[key]
        s = self.s[key]

        rho = self.rho
        eps = self.eps
        grad = param.grad.data

        h *= rho
        h += (1 - rho) * grad ** 2

        v = xp.sqrt((s + eps) / (h + eps)) * grad

        s *= rho
        s += (1 - rho) * v ** 2

        param.data -= self.lr * v

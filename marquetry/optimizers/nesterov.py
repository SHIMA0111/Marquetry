from marquetry import cuda_backend
from marquetry.optimizer import Optimizer


class Nesterov(Optimizer):
    """Nesterov Accelerated Gradient (NAG) optimizer for updating model parameters during training.

        Nesterov Accelerated Gradient is an optimization algorithm that improves upon traditional momentum
        by reducing the effect of the gradient at the current position.
        It accelerates convergence and helps avoid overshooting the optimal solution.

        Args:
            lr (float): The learning rate for updating parameters.
                Default is 0.01.
            momentum (float): The momentum factor that controls the influence of past gradients.
                Default is 0.9.

        Tip:
            When you would like to optimize your model's parameter,
            please set the model to this using ``prepare`` method.

        Examples:
            >>> optimizer = Nesterov()
            >>> model = marquetry.models.MLP([128, 256, 64, 10])
            >>> optimizer.prepare(model)
            >>> optimizer.update()

    """

    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()

        self.lr = lr
        self.momentum = momentum

        self.v = {}

    def _update_one(self, param):
        key = id(param)

        xp = cuda_backend.get_array_module(param.data)

        if key not in self.v:
            self.v[key] = xp.zeros_like(param.data)

        v = self.v[key]
        momentum = self.momentum
        grad = param.grad.data

        v *= momentum
        v -= self.lr * grad

        param.data += momentum * momentum * v
        param.data -= (1 + momentum) * self.lr * grad

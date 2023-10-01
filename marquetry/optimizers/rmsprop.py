from marquetry import cuda_backend
from marquetry.optimizer import Optimizer


class RMSProp(Optimizer):
    """Root Mean Square Propagation(RMSProp) optimizer for updating model parameters during training.

        RMSProp is an optimization algorithm that adapts the learning rates for each parameter
        based on the past gradients.
        It helps prevent the learning rate from being too large, which can lead to divergence.

        The `decay` parameter controls the rate of decay for the moving average.

        Args:
            learning_rate (float): The learning rate for updating parameters.
                Default is 0.01.
            decay (float): The decay rate for the moving average of past gradients.
                Default is 0.99.
            eps (float): A small constant added to the denominator to prevent division by zero.
                Default is 1e-8.

        Tip:
            When you would like to optimize your model's parameter,
            please set the model to this using ``prepare`` method.

        Examples:
            >>> optimizer = RMSProp()
            >>> model = marquetry.models.MLP([128, 256, 64, 10])
            >>> optimizer.prepare(model)
            >>> optimizer.update()

    """

    def __init__(self, learning_rate=0.01, decay=0.99, eps=1e-8):
        super().__init__()
        self.lr = learning_rate
        self.decay = decay
        self.eps = eps

        self.histories = {}

    def _update_one(self, param):
        h_key = id(param)

        xp = cuda_backend.get_array_module(param.data)
        if h_key not in self.histories:
            self.histories[h_key] = xp.zeros_like(param.data)

        history = self.histories[h_key]
        grad = param.grad.data

        history *= self.decay
        history += (1 - self.decay) * grad ** 2

        param.data -= self.lr * grad / (xp.sqrt(history) + self.eps)

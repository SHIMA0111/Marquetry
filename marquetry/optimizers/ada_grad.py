from marquetry import cuda_backend
from marquetry.optimizer import Optimizer


class AdaGrad(Optimizer):
    """Adaptive Gradient Algorithm(AdaGrad) optimizer for updating model parameters during training.

        AdaGrad is an optimization algorithm that adapts the learning rate for each parameter
        individually based on historical gradient information.
        It is well-suited for problems with sparse gradients.
        The learning rate is adjusted based on the historical gradient information.

        Args:
            learning_rate (float): The learning rate for updating parameters.
                Default is 0.001.
            eps (float): A small value (epsilon) added to the denominator to prevent division by zero.
                Default is 1e-8.

        Tip:
            When you would like to optimize your model's parameter,
            please set the model to this using ``prepare`` method.

        Examples:
            >>> optimizer = AdaGrad()
            >>> model = marquetry.models.MLP([128, 256, 64, 10])
            >>> optimizer.prepare(model)
            >>> optimizer.update()

    """

    def __init__(self, learning_rate=0.001, eps=1e-8):
        super().__init__()
        self.lr = learning_rate
        self.eps = eps

        self.histories = {}

    def _update_one(self, param):
        h_key = id(param)

        xp = cuda_backend.get_array_module(param.data)
        if h_key not in self.histories:
            self.histories[h_key] = xp.zeros_like(param.data)

        history = self.histories[h_key]
        grad = param.grad.data

        history += grad ** 2
        param.data -= self.lr * grad / (xp.sqrt(history) + self.eps)

from marquetry.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer for updating model parameters during training.

        SGD is an optimization algorithm that updates model parameters using a single training sample at a time.

        It is suitable for training large datasets and is a fundamental optimization technique in machine learning.
        The learning rate controls the step size for parameter updates.

        Args:
            learning_rate (float): The learning rate for updating parameters.

        Tip:
            When you would like to optimize your model's parameter,
            please set the model to this using ``prepare`` method.

        Examples:
            >>> optimizer = SGD()
            >>> model = marquetry.models.MLP([128, 256, 64, 10])
            >>> optimizer.prepare(model)
            >>> optimizer.update()

    """

    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.lr = learning_rate

    def _update_one(self, param):
        param.data -= self.lr * param.grad.data

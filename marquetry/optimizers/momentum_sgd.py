from marquetry import cuda_backend
from marquetry.optimizer import Optimizer


class MomentumSGD(Optimizer):
    """Momentum Stochastic Gradient Descent (Momentum SGD) optimizer for updating model parameters during training.

        Momentum SGD is an optimization algorithm that uses a moving average of gradients (momentum) to adaptively
        adjust the learning rates for each parameter.
        It helps accelerate convergence and avoid locally optimal solution.

        This algorithm was inspired by momentum in physics.

        Args:
            learning_rate (float): The learning rate for updating parameters.
                Default is 0.01.
            decay (float): The momentum decay factor, controlling the influence of past gradients.
                Default is 0.9.

        Tip:
            When you would like to optimize your model's parameter,
            please set the model to this using ``prepare`` method.

        Examples:
            >>> optimizer = MomentumSGD()
            >>> model = marquetry.models.MLP([128, 256, 64, 10])
            >>> optimizer.prepare(model)
            >>> optimizer.update()

    """

    def __init__(self, learning_rate=0.01, decay=0.9):
        super().__init__()
        self.lr = learning_rate
        self.momentum = decay

        self.momentum_vector = {}

    def _update_one(self, param):
        v_key = id(param)

        if v_key not in self.momentum_vector:
            xp = cuda_backend.get_array_module(param.data)
            self.momentum_vector[v_key] = xp.zeros_like(param.data)

        pre_vector = self.momentum_vector[v_key]
        pre_vector *= self.momentum
        pre_vector -= (1 - self.momentum) * param.grad.data

        param.data += pre_vector


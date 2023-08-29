from marquetry import cuda_backend
from marquetry.optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.01, decay=0.99, eps=1e-8):
        super().__init__()
        self.lr = learning_rate
        self.decay = decay
        self.eps = eps

        self.histories = {}

    def update_one(self, param):
        h_key = id(param)

        xp = cuda_backend.get_array_module(param.data)
        if h_key not in self.histories:
            self.histories[h_key] = xp.zeros_like(param.data)

        history = self.histories[h_key]
        grad = param.grad.data

        history *= self.decay
        history += (1 - self.decay) * grad ** 2

        param.data -= self.lr * grad / (xp.sqrt(history) + self.eps)


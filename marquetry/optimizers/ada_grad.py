from marquetry import optimizer, cuda_backend


class AdaGrad(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, eps=1e-8):
        super().__init__()
        self.lr = learning_rate
        self.eps = eps

        self.histories = {}

    def update_one(self, param):
        h_key = id(param)

        xp = cuda_backend.get_array_module(param.data)
        if h_key not in self.histories:
            self.histories[h_key] = xp.zeros_like(param.data)

        history = self.histories[h_key]
        grad = param.grad.data

        history += grad ** 2
        param.data -= self.lr * grad / (xp.sqrt(history) + self.eps)

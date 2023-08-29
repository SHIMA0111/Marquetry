from marquetry.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.lr = learning_rate

    def update_one(self, param):
        param.data -= self.lr * param.grad.data

from marquetry import optimizer, cuda_backend


class MomentumSGD(optimizer.Optimizer):
    def __init__(self, learning_rate=0.01, decay=0.9):
        super().__init__()
        self.lr = learning_rate
        self.momentum = decay

        self.momentum_vector = {}

    def update_one(self, param):
        v_key = id(param)

        if v_key not in self.momentum_vector:
            xp = cuda_backend.get_array_module(param.data)
            self.momentum_vector[v_key] = xp.zeros_like(param.data)

        pre_vector = self.momentum_vector[v_key]
        pre_vector *= self.momentum
        pre_vector -= (1 - self.momentum) * param.grad.data

        param.data += pre_vector


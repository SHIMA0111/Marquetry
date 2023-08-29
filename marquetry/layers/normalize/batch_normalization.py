import marquetry.cuda_backend as cuda_backend
from marquetry import functions
from marquetry import Layer
from marquetry import Parameter


class BatchNormalization(Layer):
    def __init__(self, decay=0.9):
        super().__init__()
        self.avg_mean = Parameter(None, name="avg_mean")
        self.avg_var = Parameter(None, name="avg_var")
        self.gamma = Parameter(None, name="gamma")
        self.beta = Parameter(None, name="beta")

        self.decay = decay

    def __call__(self, x):
        xp = cuda_backend.get_array_module(x)
        if self.avg_mean.data is None:
            input_shape = x.shape[1]
            if self.avg_mean.data is None:
                self.avg_mean.data = xp.zeros(input_shape, dtype=x.dtype)
            if self.avg_var.data is None:
                self.avg_var.data = xp.ones(input_shape, dtype=x.dtype)
            if self.gamma.data is None:
                self.gamma.data = xp.ones(input_shape, dtype=x.dtype)
            if self.beta.data is None:
                self.beta.data = xp.zeros(input_shape, dtype=x.dtype)

        return functions.batch_normalization(x, self.gamma, self.beta, self.avg_mean.data, self.avg_var.data, self.decay)

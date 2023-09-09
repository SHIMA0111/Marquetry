import marquetry.cuda_backend as cuda_backend
from marquetry import functions
from marquetry import Layer
from marquetry import Parameter


class BatchNormalization(Layer):
    """Batch normalization layer.

        Batch normalization is a technique used in deep neural networks to stabilize and accelerate training.
        This layer normalizes the input data by using normalization and adjustable parameters (gamma and beta).
        It helps prevent issues like vanishing gradients, reduces co-variate shift,
        and allows for faster convergence.

        Args:
            decay (float): The weighting factor for the moving averages of mean and variance.
                A smaller value will make the moving averages change more slowly.
                Default is 0.9.

        Attributes:
            gamma (marquetry.Parameter): The gamma parameter used for scaling.
            beta (marquetry.Parameter): The beta parameter used for shifting.
            avg_mean (marquetry.Parameter): The moving average of the mean.
            avg_var (marquetry.Parameter): The moving average of the variance.

        Examples:
            >>> x = np.random.rand(4, 5)
            >>> x.mean(axis=0)
            array([0.34748032, 0.55353551, 0.46282959, 0.4974482 , 0.60365538])
            >>> x.std(axis=0)
            array([0.25182043, 0.25240281, 0.29050406, 0.37605534, 0.30533898])
            >>> batch_norm = BatchNormalization()
            >>> y = batch_norm(x)
            >>> y.data.mean(axis=0)
            array([0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
            (Depending on the computing environment, the mean is not 0.0 but the gap is very small
            (generally under 1e-15).)
            >>> y.data.std(axis=0)
            array([1., 1., 1., 1., 1.])


    """

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

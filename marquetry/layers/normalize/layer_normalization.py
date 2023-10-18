from marquetry import cuda_backend
from marquetry import functions
from marquetry import Layer
from marquetry import Parameter


class LayerNormalization(Layer):
    """Layer normalization layer.

        Layer normalization is a technique used in deep neural networks to stabilize and accelerate training.
        It normalizes the input data by normalization and  adjustable parameters called gamma(scale factor) and
        beta(shift factor).
        This doesn't depend on the batch size due to normalize by the only 1 layer values.
        In some use cases, the stability is better than :class:`marquetry.functions.BatchNormalization`.
        This helps in preventing issues like vanishing gradients, reduce co-variate shift,
        and allow for faster convergence.

        Args:
            eps (float): A small constant value preventing zero-division.
                Default is 1e-15.

        Attributes:
            gamma (marquetry.Parameter): The gamma parameter used for scaling.
            beta (marquetry.Parameter): The beta parameter used for shifting.

        Examples:
            >>> x = np.random.rand(4, 5)
            >>> x.mean(axis=1)
            array([0.39610748, 0.30893099, 0.2925813 , 0.49962495])
            >>> x.std(axis=1)
            array([0.24654998, 0.26883532, 0.22135444, 0.2950873 ])
            >>> layer_norm = LayerNormalization()
            >>> y = layer_norm(x)
            >>> y.data.mean(axis=1)
            array([0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
            (Depending on the computing environment, the mean is not 0.0 but the gap is very small, generally under 1e-15).)
            >>> y.data.std(axis=1)
            array([1., 1., 1., 1.])

    """

    def __init__(self, eps=1e-15):
        super().__init__()

        self.eps = eps

        self.gamma = Parameter(None, name="gamma")
        self.beta = Parameter(None, name="beta")

    def __call__(self, x):
        xp = cuda_backend.get_array_module(x)

        if self.gamma.data is None:
            input_shape = int(x.size / x.shape[0])
            self.gamma.data = xp.ones(input_shape, dtype=x.dtype)
            self.beta.data = xp.zeros(input_shape, dtype=x.dtype)

        return functions.layer_normalization(x, self.gamma, self.beta, self.eps)

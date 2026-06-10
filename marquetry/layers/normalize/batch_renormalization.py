import marquetry.cuda_backend as cuda_backend
from marquetry import functions
from marquetry import Layer
from marquetry import Parameter


class BatchRenormalization(Layer):
    """Batch renormalization layer.

        Batch renormalization extends batch normalization so that train-time normalization
        stays close to the inference-time normalization even with small or non-i.i.d.
        mini-batches. The batch statistics are corrected toward the running statistics
        by the clipped factors ``r`` and ``d``.

        With ``rmax=1`` and ``dmax=0`` (default), the behavior is identical to
        :class:`marquetry.layers.BatchNormalization`. The paper suggests starting
        the training with the defaults and gradually relaxing the bounds
        (e.g. ``rmax`` toward 3 and ``dmax`` toward 5).

        Args:
            rmax (float): The clipping bound of the std correction factor ``r``.
                Default is 1.0 (no correction).
            dmax (float): The clipping bound of the mean correction factor ``d``.
                Default is 0.0 (no correction).
            decay (float): The weighting factor for the moving averages of mean and variance.
                Default is 0.9.
            eps (float): A small constant value preventing zero-division.
                Default is 1e-15.

        Attributes:
            gamma (marquetry.Parameter): The gamma parameter used for scaling.
            beta (marquetry.Parameter): The beta parameter used for shifting.
            avg_mean (marquetry.Parameter): The moving average of the mean.
            avg_var (marquetry.Parameter): The moving average of the variance.

        References:
            Batch Renormalization: Towards Reducing Minibatch Dependence
            in Batch-Normalized Models (https://arxiv.org/abs/1702.03275)
    """

    def __init__(self, rmax=1.0, dmax=0.0, decay=0.9, eps=1e-15):
        super().__init__()

        self.rmax = rmax
        self.dmax = dmax
        self.decay = decay
        self.eps = eps

        self.avg_mean = Parameter(None, name="avg_mean")
        self.avg_var = Parameter(None, name="avg_var")
        self.gamma = Parameter(None, name="gamma")
        self.beta = Parameter(None, name="beta")

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

        return functions.batch_renormalization(x, self.gamma, self.beta, self.avg_mean.data,
                                               self.avg_var.data, self.rmax, self.dmax,
                                               self.decay, self.eps)

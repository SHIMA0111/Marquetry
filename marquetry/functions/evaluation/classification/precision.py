from marquetry import cuda_backend
from marquetry import Function


class Precision(Function):
    """Compute precision based on true labels and predicted values using a specified threshold.

        This class defines a function that computes precision based on the true binary labels and predicted values.
        The precision indicates the accuracy of the prediction, more easily, this evaluator indicates
        how many correct the positive prediction compare the all positive prediction.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, threshold, dtype="float64"):
        if not 0. <= threshold <= 1.:
            raise ValueError("threshold should be in (0.0, 1.0), but got {}".format(threshold))
        self.threshold = threshold
        self.dtype = dtype

    def forward(self, y, t):
        xp = cuda_backend.get_array_module(y)

        assert len(xp.unique(t)) <= 2

        pred = xp.asarray((y >= self.threshold), dtype="f").reshape(t.shape)

        pred_positive_num = pred.sum()
        true_positive_num = pred[t == 1].sum()

        self.retain_inputs(())
        if pred_positive_num == 0:
            return xp.asarray(0.0, dtype=self.dtype)
        else:
            return xp.asarray(true_positive_num / pred_positive_num, dtype=self.dtype)


def precision(y, t, threshold=0.7, dtype="float64"):
    """Compute precision based on true labels and predicted values using a specified threshold.

        This function defines that computes precision based on the true binary labels and predicted values.
        The precision indicates the accuracy of the prediction, more easily, this evaluator indicates
        how many correct the positive prediction compare the all positive prediction.

        Precision is calculated as:
            :math:`Precision = TP / TP + FP`
            (TP -> TruePositive, FP -> FalsePositive)

        Args:
            y (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted values.
            t (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The true binary labels.
            threshold (float): The threshold value used to determine binary predictions.
                Defaults to 0.7.
            dtype (str or numpy.dtype):
                The return value's dtype, Default is "float64" that means 64-bit float value.

        Returns:
            :class:`marquetry.Container`: The precision based on the true binary labels
                and predicted values using the specified threshold.
    """

    return Precision(threshold, dtype)(y, t)


class MultiPrecision(Function):
    """Calculate precision for a specific target class based on true labels and predicted values.

        This class defines a function that calculates precision for a specific target class based on
        the true labels and predicted values.
        Precision measures the accuracy of positive predictions for the specified target class,
        indicating how many positive predictions are correct compare the all positive prediction.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """

    def __init__(self, target_class, dtype="float64"):
        self.target_class = target_class
        self.dtype = dtype

    def forward(self, y, t):
        xp = cuda_backend.get_array_module(y)

        assert len(xp.unique(t)) > 2

        if y.size != t.size:
            pred = xp.argmax(y, axis=1).reshape(t.shape)
        else:
            pred = y

        pred_positive = xp.asarray(pred == self.target_class, dtype="f")
        pred_true_map = xp.asarray(pred == t, dtype="f")

        pred_true_positive = (pred_positive * pred_true_map).sum()
        pred_positive_num = pred_positive.sum()

        self.retain_inputs(())
        if pred_positive_num == 0:
            return xp.asarray(0.0, dtype=self.dtype)
        else:
            return xp.asarray(pred_true_positive / pred_positive_num, dtype=self.dtype)


def multi_precision(y, t, target_class=1, dtype="float64"):
    """Calculate precision for a specific target class based on true labels and predicted values.


        This function defines that calculates precision for a specific target class based on
        the true labels and predicted values.
        Precision measures the accuracy of positive predictions for the specified target class, indicating how many
        positive predictions are correct compare the all positive prediction.

        Precision is calculated as:
            :math:`Precision = TP / TP + FP`
            (TP -> True Positives, FP -> False Positives)

        Args:
            y (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted values.
            t (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The true labels.
            target_class (int): The class index to calculate precision for.
                Defaults to 1.
            dtype (str or numpy.dtype):
                The return value's dtype, Default is "float64" that means 64-bit float value.

        Returns:
            :class:`marquetry.Container`:
                The precision for the specified target class based on the true labels and predicted values.
    """

    return MultiPrecision(target_class, dtype)(y, t)

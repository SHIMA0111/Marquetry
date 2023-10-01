import typing

from marquetry import cuda_backend
from marquetry import Function


class ClassificationError(Function):
    """Compute classification error based on true labels and predictions.

        This class defines that computes the classification error based on the true labels and predicted labels.
        It supports an optional `ignore_label` parameter to exclude certain labels from the error calculation.

        Note:
            Generally, you don't need to execute ``forward`` and ``backward`` method manually.
            You should use only ``__call__`` method.
    """
    def __init__(self, ignore_label, dtype="float64"):
        self.ignore_label = ignore_label
        self.dtype = dtype

    def forward(self, y, t):
        xp = cuda_backend.get_array_module(y)

        self.retain_inputs(())
        if self.ignore_label is not None:
            mask = xp.asarray(t == self.ignore_label).astype("f")
            ignore_cnt = mask.sum()

            pred = xp.where(mask, self.ignore_label, y.argmax(axis=1).reshape(t.shape))

            count = t.size - xp.asarray(pred == t).sum()
            total = t.size - ignore_cnt

            if total == 0:
                return xp.asarray(0.0, dtype=self.dtype)
            else:
                return xp.asarray(float(count) / total, dtype=self.dtype)

        else:
            if y.size != t.size:
                pred = y.argmax(axis=1).reshape(t.shape)
            else:
                pred = y

            return xp.asarray((pred != t)).mean(dtype=self.dtype)


def classification_error(y, t, ignore_label: typing.Optional[int] = None, dtype="float64"):
    """Compute classification error based on true labels and predictions.


        This function defines that computes the classification error based on the true labels and predicted labels.
        It supports an optional `ignore_label` parameter to exclude certain labels from the error calculation.

        ClassificationError is calculated as:
            :math:`ClassificationError = (FP + FN) / (TP + FP + TN + FN)`

            (TP -> TruePositive, FP -> FalsePositive, TN -> TrueNegative, FN -> FalseNegative)


        Args:
            y (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted labels.
            t (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The true labels.
            ignore_label (int or None):
                The label value to be ignored when calculating the classification error.
                If None, no labels are ignored.
            dtype (str or numpy.dtype):
                The return value's dtype, Default is "float64" that means 64-bit float value.

        Returns:
            :class:`marquetry.Container`: The classification error based on the true
                labels and predicted labels, excluding labels specified by 'ignore_label' (if any).
        """

    return ClassificationError(ignore_label, dtype)(y, t)


class BinaryClassificationError(Function):
    """Compute binary classification error based on true labels and predicted values.

        This class defines a function that computes the binary classification error based on the true binary labels and
        predicted values. The binary classification error is calculated based on a specified threshold.

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

        pred = xp.asarray((y >= self.threshold), dtype="f")

        self.retain_inputs(())
        return xp.asarray((pred != t)).mean(dtype=self.dtype)


def binary_classification_error(y, t, threshold: float = 0.7, dtype="float64"):
    """Compute binary classification error based on true labels and predicted values.


        This function defines that computes the binary classification error based on the true binary labels and
        predicted values. The binary classification error is calculated based on a specified threshold.

        ClassificationError is calculated as:
            :math:`ClassificationError = (FP + FN) / (TP + FP + TN + FN)`

            (TP -> TruePositive, FP -> FalsePositive, TN -> TrueNegative, FN -> FalseNegative)

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
            :class:`marquetry.Container`: The binary classification error based on the true binary labels and
                predicted values using the specified threshold.
    """
    return BinaryClassificationError(threshold, dtype)(y, t)

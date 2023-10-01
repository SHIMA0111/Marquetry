from marquetry import cuda_backend
from marquetry import Function


class Recall(Function):
    """Calculate recall based on true labels and predicted values using a specified threshold.

        This class defines a function that calculates recall based on the true binary labels and predicted values.
        Recall measures the ability to correctly identify positive cases,
        indicating how many actual positive labels were correctly predicted.

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

        t_positive_num = xp.asarray((t == 1), dtype="f").sum()
        true_positive_num = pred[t == 1].sum()

        self.retain_inputs(())

        if t_positive_num == 0:
            return xp.asarray(0.0, dtype=self.dtype)
        else:
            return xp.asarray(true_positive_num / t_positive_num, dtype=self.dtype)


def recall(y, t, threshold=0.7, dtype="float64"):
    """Calculate recall based on true labels and predicted values using a specified threshold.

        This function defines that calculates recall based on the true binary labels and predicted values.
        Recall measures the ability to correctly identify positive cases,
        indicating how many actual positive labels were correctly predicted.

        Recall is calculated as:
            :math:`Recall = TP / (TP + FN)`
            (TP -> True Positives, FN -> False Negatives)

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
            :class:`marquetry.Container`: The recall based on the true binary labels
                and predicted values using the specified threshold.
    """

    return Recall(threshold, dtype)(y, t)


class MultiRecall(Function):
    """Calculate recall for a specific target class based on true labels and predicted values.

        This class defines a function that calculates recall
        for a specific target class based on the true labels and predicted values.
        Recall measures the ability to correctly identify positive cases for the specified target class,
        indicating how many actual positive labels for that class were correctly predicted.

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

        t_positive = xp.asarray((t == self.target_class), dtype="f")

        pred_positive = xp.asarray((pred == self.target_class), dtype="f")
        true_positive_num = (pred_positive * t_positive).sum()
        t_positive_num = t_positive.sum()

        self.retain_inputs(())
        if t_positive_num == 0:
            return xp.asarray(0.0, dtype=self.dtype)
        else:
            return xp.asarray(true_positive_num / t_positive_num, dtype=self.dtype)


def multi_recall(y, t, target_class=1, dtype="float64"):
    """Calculate recall for a specific target class based on true labels and predicted values.

        This function defines that calculates recall
        for a specific target class based on the true labels and predicted values.
        Recall measures the ability to correctly identify positive cases for the specified target class,
        indicating how many actual positive labels for that class were correctly predicted.

        Recall is calculated as:
            :math:`Recall = TP / (TP + FN)`
            (TP -> True Positives, FN -> False Negatives)

        Args:
            y (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted values.
            t (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The true labels.
            target_class (int): The class index to calculate recall for.
                Defaults to 1.
            dtype (str or numpy.dtype):
                The return value's dtype, Default is "float64" that means 64-bit float value.

        Returns:
            :class:`marquetry.Container`: The recall for the specified target class
                based on the true labels and predicted values.
    """

    return MultiRecall(target_class, dtype)(y, t)

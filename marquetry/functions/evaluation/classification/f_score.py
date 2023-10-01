from marquetry import cuda_backend
from marquetry import Function


class FScore(Function):
    """Compute the F-score based on true labels and predicted values using a specified threshold.

        This class defines a function that computes the F-score based on the true binary labels and predicted values.

        The F-score is a way of combining the precision and recall of the model,
        and it is defined as the harmonic mean of the model’s precision and recall.

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

        true_positive_num = pred[t == 1].sum()
        pred_positive_num = pred.sum()
        target_positive_num = xp.asarray((t == 1), dtype="f").sum()

        precision_value, recall_value = _precision_recall_validator(true_positive_num, pred_positive_num,
                                                                    target_positive_num, xp=xp)

        self.retain_inputs(())
        if precision_value == 0. and recall_value == 0.:
            return xp.asarray(0.0, dtype=self.dtype)
        else:
            f_score_value = 2 * precision_value * recall_value / (precision_value + recall_value)
            return xp.asarray(f_score_value.data, dtype=self.dtype)


def f_score(y, t, threshold=0.7, dtype="float64"):
    """Compute the F-score based on true labels and predicted values using a specified threshold.

        This function defines that computes the F-score based on the true binary labels and predicted values.

        The F-score is a way of combining the precision and recall of the model,
        and it is defined as the harmonic mean of the model’s precision and recall.

        F-score is calculated as:
            :math:`F-score = (2 * precision * recall) / (precision + recall)`

        More species, please see :meth:`marquetry.functions.precision` and :meth:`marquetry.functions.recall`.

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
            :class:`marquetry.Container`: The F-score based on the true binary labels
                and predicted values using the specified threshold.
    """

    return FScore(threshold, dtype)(y, t)


class MultiFScore(Function):
    """Compute the F-score for a specific target class based on true labels and predicted values.

        This class defines a function that computes the F-score for a specific target class based on
        the true labels and predicted values.

        The F-score is a way of combining the precision and recall of the model,
        and it is defined as the harmonic mean of the model’s precision and recall.

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
        true_map = xp.asarray(pred == t, dtype="f")

        true_positive_num = xp.asarray(pred_positive * true_map, dtype="f").sum()
        pred_positive_num = pred_positive.sum()
        target_positive_num = xp.asarray(t == self.target_class, dtype="f").sum()

        precision_value, recall_value = _precision_recall_validator(true_positive_num, pred_positive_num,
                                                                    target_positive_num, xp=xp)

        self.retain_inputs(())
        if precision_value == 0. and recall_value == 0.:
            return xp.asarray(0.0, dtype=self.dtype)
        else:
            f_score_value = 2 * precision_value * recall_value / (precision_value + recall_value)
            return xp.asarray(f_score_value.data, dtype=self.dtype)


def multi_f_score(y, t, target_class=1, dtype="float64"):
    """Compute the F-score for a specific target class based on true labels and predicted values.

        This function defines that computes the F-score for a specific target class
        based on the true labels and predicted values.
        The F-score is a way of combining the precision and recall of the model,
        and it is defined as the harmonic mean of the model’s precision and recall.

        F-score is calculated as:
            :math:`F-score = (2 * precision * recall) / (precision + recall)`

        More species,
        please see :meth:`marquetry.functions.multi_precision` and :meth:`marquetry.functions.multi_recall`.

        Args:
            y (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted values.
            t (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The true labels.
            target_class (int):
                The class index to be treated as the positive class when calculating the F-score.
                Defaults to 1.
            dtype (str or numpy.dtype):
                The return value's dtype, Default is "float64" that means 64-bit float value.

        Returns:
            :class:`marquetry.Container`: The F-score for the specified target class
                based on the true labels and predicted values.
    """
    return MultiFScore(target_class, dtype)(y, t)


def _precision_recall_validator(true_positive_num, pred_positive_num, target_positive_num, xp):
    """Validate pred_positive_num and target_positive_num to avoid Zero-Division and calc the precision and recall.

        Args:
            true_positive_num (int): The true positive num means a model predicts a ``True``,
                and it is actuary ``True`` counts.
            pred_positive_num (int): The true pred num means a model predicts a ``True`` counts.
            target_positive_num (int): The real true num means a target label's ``True`` counts.

        Returns:
            :class:`cupy.ndarray` or :class:`numpy.ndarray`: The precision and recall value.
    """
    if pred_positive_num == 0 and target_positive_num == 0:
        precision_value = xp.asarray(0.0)
        recall_value = xp.asarray(0.0)
    elif pred_positive_num == 0:
        precision_value = xp.asarray(0.0)
        recall_value = xp.asarray(true_positive_num / target_positive_num)
    elif target_positive_num == 0:
        precision_value = xp.asarray(true_positive_num / pred_positive_num)
        recall_value = xp.asarray(0.0)
    else:
        precision_value = xp.asarray(true_positive_num / pred_positive_num)
        recall_value = xp.asarray(true_positive_num / target_positive_num)

    return precision_value, recall_value

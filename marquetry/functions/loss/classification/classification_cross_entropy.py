from marquetry import functions


def classification_cross_entropy(x, t):
    """Calculate the Cross-Entropy loss for classification tasks based on the dimensionality of the input.

        This function calculates the Cross-Entropy loss for classification problems,
        depending on the dimensionality of the input data.

        If the input has a single dimension or a single output unit,
        it calculates the :class:`marquetry.functions.sigmoid_cross_entropy` loss.

        If the input has multiple dimensions or output units,
        it calculates the :class:`marquetry.functions.softmax_cross_entropy` loss.

        Args:
            x (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The predicted values.
            t (:class:`marquetry.Container`, :class:`numpy.ndarray`, or :class:`cupy.ndarray`):
                The true labels or binary targets.

        Returns:
            :class:`marquetry.Container`: The Cross-Entropy loss calculated
                based on the dimensionality of the input data.
    """

    if x.ndim == 1 or x.shape[1] == 1:
        return functions.sigmoid_cross_entropy(x, t)
    else:
        return functions.softmax_cross_entropy(x, t)

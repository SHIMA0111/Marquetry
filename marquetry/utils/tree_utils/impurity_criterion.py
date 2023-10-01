from marquetry import cuda_backend


def impurity_criterion(target, criterion=None, target_type="classification"):
    """Compute an impurity criterion for a given target container.

        This function computes an impurity criterion for a given target container
        based on the specified criteria and target type.

        Args:
            target (:class:`numpy.ndarray`, or :class:`cupy.ndarray`):
                The target array for which the impurity criterion should be computed.
            criterion (str or None): The impurity criterion to use.
                For classification, options are "gini" or "entropy."
                For regression, options are "rss" (Residual Sum of Squares) or "mae" (Mean Absolute Error).
                Default is None, which selects the appropriate criterion based on the target_type.
            target_type (str): The type of the target container.
                Options are "classification" or "regression" Default is "classification".

        Returns:
            float: The computed impurity criterion.
    """

    if target_type == "classification":
        criterion = criterion if criterion is not None else "gini"
        return _classification_impurity_criterion(target, criterion)
    elif target_type == "regression":
        criterion = criterion if criterion is not None else "rss"
        return _regression_impurity_criterion(target, criterion)
    else:
        raise ValueError("target_type: {} is not supported.".format(target))


def _classification_impurity_criterion(target, criterion):
    """Compute a classification impurity criterion for a given target container.

        This function computes a classification impurity criterion for a given target container
        based on the specified criterion.

        Args:
            target (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The target container for which the impurity criterion should be computed.
            criterion (str): The impurity criterion to use.
                Options are "gini" or "entropy".

        Returns:
            float: The computed classification impurity criterion.

    """

    xp = cuda_backend.get_array_module(target)
    classes = xp.unique(target)
    num_samples = len(target)

    if criterion.lower() == "gini":
        result = 1.
        for class_num in classes:
            # calc each class rate
            rate = float(len(target[target == class_num])) / num_samples
            result -= rate ** 2

    elif criterion.lower() == "entropy":
        result = 0.
        for class_num in classes:
            # calc each class rate
            rate = float(len(target[target == class_num])) / num_samples
            result -= rate * xp.log2(rate)
    else:
        raise ValueError("{} is not supported as criterion.".format(criterion))

    return result


def _regression_impurity_criterion(target, criterion):
    """Compute a regression impurity criterion for a given target container.

        This function computes a regression impurity criterion for a given target container
        based on the specified criterion.

        Args:
            target (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The target container for which the impurity criterion should be computed.
            criterion (str): The impurity criterion to use.
                Options are "rss" (Residual Sum of Squares) or "mae" (Mean Absolute Error).

        Returns:
            float: The computed regression impurity criterion.
    """

    xp = cuda_backend.get_array_module(target)
    estimate_target = target.mean()
    if criterion.lower() == "rss":
        result = xp.sum((target - estimate_target) ** 2)
    elif criterion.lower() == "mae":
        result = xp.mean(xp.abs(target - estimate_target))
    else:
        raise ValueError("{} is not supported as criterion.".format(criterion))

    return result

import marquetry


def information_gain(target, target_left, target_right, criterion=None, target_type="classification"):
    """Compute the information gain from splitting a dataset into two subsets.

        Information gain measures how much the impurity decreases from before splitting
        to after splitting a dataset into two subsets.
        It is commonly used in decision tree algorithms to evaluate the quality of a split.

        Args:
            target (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The target container before splitting.
            target_left (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The target container of the left subset after splitting.
            target_right (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The target container of the right subset after splitting.
            criterion (str or None): The impurity criterion to use.
                For classification, options are "gini" or "entropy."
                For regression, options are "rss" (Residual Sum of Squares) or "mae" (Mean Absolute Error).
                Default is None, which selects the appropriate criterion based on the target_type.
            target_type (str): The type of the target container.
                Options are "classification" or "regression."
                Default is "classification."

        Returns:
            float: The computed information gain.

    """
    impurity_target = marquetry.utils.impurity_criterion(target, criterion=criterion, target_type=target_type)
    impurity_left = marquetry.utils.impurity_criterion(target_left, criterion=criterion, target_type=target_type)
    impurity_right = marquetry.utils.impurity_criterion(target_right, criterion=criterion, target_type=target_type)

    split_mean_impurity = (float(len(target_left) / len(target)) * impurity_left +
                           float(len(target_right) / len(target) * impurity_right))
    info_gain = impurity_target - split_mean_impurity

    return info_gain

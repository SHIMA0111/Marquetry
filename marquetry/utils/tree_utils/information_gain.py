import marquetry


def information_gain(target, target_left, target_right, criterion=None, target_type="classification"):
    """
    information_gain indicates how much cleansing the impurity from before splitting to after.
    """
    impurity_target = marquetry.utils.impurity_criterion(target, criterion=criterion, target_type=target_type)
    impurity_left = marquetry.utils.impurity_criterion(target_left, criterion=criterion, target_type=target_type)
    impurity_right = marquetry.utils.impurity_criterion(target_right, criterion=criterion, target_type=target_type)

    split_mean_impurity = (float(len(target_left) / len(target)) * impurity_left +
                           float(len(target_right) / len(target) * impurity_right))
    info_gain = impurity_target - split_mean_impurity

    return info_gain

from marquetry import cuda_backend


def impurity_criterion(target, criterion=None, target_type="classification"):
    if target_type == "classification":
        return _classification_impurity_criterion(target, criterion)
    elif target_type == "regression":
        return _regression_impurity_criterion(target, criterion)
    else:
        raise ValueError("target_type: {} is not supported.".format(target))


def _classification_impurity_criterion(target, criterion="gini"):
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


def _regression_impurity_criterion(target, criterion="rss"):
    xp = cuda_backend.get_array_module(target)
    estimate_target = target.mean()
    if criterion.lower() == "rss":
        result = xp.sum((target - estimate_target) ** 2)
    elif criterion.lower() == "mae":
        result = xp.mean(xp.abs(target - estimate_target))
    else:
        raise ValueError("{} is not supported as criterion.".format(criterion))

    return result

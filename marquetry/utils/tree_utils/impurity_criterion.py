from marquetry import cuda_backend


def impurity_criterion(target, criterion="gini"):
    xp = cuda_backend.get_array_module(target)
    classes = xp.unique(target)
    num_samples = len(target)

    if criterion == "gini":
        result = 1.
        for class_num in classes:
            # calc each class rate
            rate = float(len(target[target == class_num])) / num_samples
            result -= rate ** 2

    elif criterion == "entropy":
        result = 0.
        for class_num in classes:
            # calc each class rate
            rate = float(len(target[target == class_num])) / num_samples
            result -= rate * xp.log2(rate)
    else:
        raise Exception("{} is not supported as criterion.".format(criterion))

    return result

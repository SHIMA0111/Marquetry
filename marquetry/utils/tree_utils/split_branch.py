import marquetry
from marquetry import cuda_backend


def split_branch(data, target, class_list, criterion="gini", seed=None, is_leaf=False):
    """
    return: is_leave, (label, impurity), feature, threshold
    """
    xp = cuda_backend.get_array_module(data)

    count_classes_datas = [len(target[target == class_num]) for class_num in class_list]

    current_impurity = marquetry.utils.impurity_criterion(target, criterion=criterion, target_type="classification")
    class_counts = dict(zip(class_list, count_classes_datas))
    label = max(class_counts.items(), key=lambda count: count[1])[0]

    if len(xp.unique(target)) == 1:
        # If target labels already have only 1 label, the impurity is 0 and, the data can't split anymore.
        return True, (label, current_impurity), None, None

    class_counts = dict(zip(class_list, count_classes_datas))
    label = max(class_counts.items(), key=lambda count: count[1])[0]

    if is_leaf:
        return True, (label, current_impurity), None, None

    num_features = data.shape[1]
    pre_info_gain = 0.0

    xp.random.seed(seed)

    shuffle_features_list = list(xp.random.permutation(num_features))

    feature_candidate, threshold_candidate = None, None
    for feature in shuffle_features_list:
        unique_in_feature = xp.unique(data[:, feature])
        threshold_point = (unique_in_feature[:-1] + unique_in_feature[1:]) / 2.

        for threshold in threshold_point:
            target_left = target[data[:, feature] <= threshold]
            target_right = target[data[:, feature] > threshold]

            info_gain = marquetry.utils.information_gain(target, target_left, target_right, criterion=criterion)

            if pre_info_gain < info_gain:
                pre_info_gain = info_gain
                feature_candidate = feature
                threshold_candidate = threshold

    if pre_info_gain == 0.:
        return True, (label, current_impurity), None, None

    return False, (label, current_impurity), feature_candidate, threshold_candidate

import marquetry
from marquetry import cuda_backend


def split_branch(data, target, class_list=None, criterion=None, seed=None, target_type="classification", is_leaf=False):
    """Split a branch of a decision tree based on the specified criteria and target type.

        This function splits a branch of a decision tree into two branches or determines it as a leaf node
        based on the specified criteria and target type.
        It can be used for both classification and regression tasks.

        Args:
            data (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The data used for splitting the branch.
            target (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                The target container associated with the data.
            class_list (list or None): A list of class labels used for classification tasks.
                Default is None.
            criterion (str): The impurity criterion to use.
                For classification, options are "gini" or "entropy".
                For regression, options are "rss" (Residual Sum of Squares) or "mae" (Mean Absolute Error).
                Default is None, which selects the appropriate criterion based on the target_type.
            seed (int or None): The random seed for shuffling features during feature selection.
                Default is None.
            target_type (str): The type of the target container. Options are "classification" or "regression".
                Default is "classification".
            is_leaf (bool): Indicates whether the branch should be determined as a leaf node.
                Default is False.

        Returns:
            tuple: A tuple containing information about the branch split or leaf determination like the below.

                - is_leaf(bool): True if the node is leaf, otherwise, the node is branch.

                - (label or value, impurity)(tuple): The answer label or value and the impurity value.

                - feature(int): The feature number of the source table.

                - threshold(float or int): The threshold splitting the data based on the feature.

    """

    if target_type == "classification":

        return _classification_split_branch(data, target, class_list, criterion, seed, is_leaf)
    elif target_type == "regression":
        return _regression_split_branch(data, target, criterion, seed, is_leaf)
    else:
        raise ValueError("target_type: {} is not supported.".format(target_type))


def _classification_split_branch(data, target, class_list, criterion="gini", seed=None, is_leaf=False):
    """Split a branch for classification tasks.

        This function splits a branch of a decision tree for classification tasks based on the specified criterion.

        Returns:
            tuple: A tuple containing information about the branch split or leaf determination.
    """

    if class_list is None:
        raise ValueError("classification splitter expects list object as class_list, but got {}".format(class_list))

    xp = cuda_backend.get_array_module(data)

    count_classes_datas = [len(target[target == class_num]) for class_num in class_list]

    current_impurity = marquetry.utils.impurity_criterion(target, criterion=criterion, target_type="classification")
    class_counts = dict(zip(class_list, count_classes_datas))
    label = max(class_counts.items(), key=lambda count: count[1])[0]

    if len(xp.unique(target)) == 1 or is_leaf:
        # If target labels already have only 1 label, the impurity is 0 and, the data can't split anymore.
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

            info_gain = marquetry.utils.information_gain(target, target_left, target_right,
                                                         criterion=criterion, target_type="classification")

            if pre_info_gain < info_gain:
                pre_info_gain = info_gain
                feature_candidate = feature
                threshold_candidate = threshold

    if pre_info_gain == 0.:
        return True, (label, current_impurity), None, None

    return False, (label, current_impurity), feature_candidate, threshold_candidate


def _regression_split_branch(data, target, criterion="rss", seed=None, is_leaf=False):
    """Split a branch for regression tasks.

        This function splits a branch of a decision tree for regression tasks based on the specified criterion.

        Returns:
            tuple: A tuple containing information about the branch split or leaf determination.
    """

    xp = cuda_backend.get_array_module(data)

    current_impurity = marquetry.utils.impurity_criterion(target, criterion=criterion, target_type="regression")
    value = target.mean()

    if is_leaf:
        return True, (value, current_impurity), None, None

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

            info_gain = marquetry.utils.information_gain(target, target_left, target_right,
                                                         criterion=criterion, target_type="regression")

            if pre_info_gain < info_gain:
                pre_info_gain = info_gain
                feature_candidate = feature
                threshold_candidate = threshold

    if pre_info_gain == 0.:
        return True, (value, current_impurity), None, None

    return False, (value, current_impurity), feature_candidate, threshold_candidate

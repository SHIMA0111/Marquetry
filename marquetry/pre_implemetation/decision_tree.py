import numpy as np


class _Node(object):
    def __init__(self, criterion: str = "gini", max_depth=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state

        self.depth = None
        self.label = None
        self.error = None
        self.info_gain = None
        self.feature = None
        self.threshold = None
        self.left_branch = None
        self.right_branch = None
        self.num_data_per_class = None
        self.num_samples = None

    def _criterion_calc(self, target):
        classes = np.unique(target)
        num_data = len(target)

        if self.criterion == "gini":
            val = 1
            for class_num in classes:
                rate = float(len(target[target == class_num])) / num_data
                val -= rate ** 2.

        elif self.criterion == "entropy":
            val = 0
            for class_num in classes:
                rate = float(len(target[target == class_num])) / num_data

                if rate != 0.:
                    val -= rate * np.log2(rate)

        else:
            raise Exception("You can use 'gini' or 'entropy' as indicator, but you input {}.".format(self.criterion))

        return val

    def _info_gain_criterion(self, target_origin, target_left, target_right):
        criterion_value_origin = self._criterion_calc(target_origin)
        criterion_value_left = self._criterion_calc(target_left)
        criterion_value_right = self._criterion_calc(target_right)

        mean_criterion_data = ((len(target_left) / float(len(target_origin)) * criterion_value_left) +
                               (len(target_right) / float(len(target_origin)) * criterion_value_right))

        info_gain = criterion_value_origin - mean_criterion_data

        return info_gain

    def create_threshold(self, x, t, depth, class_list):
        self.depth = depth

        self.num_samples = len(t)
        self.num_data_per_class = [len(t[t == class_num]) for class_num in class_list]

        if len(np.unique(t)) == 1:
            self.label = t[0]
            self.error = self._criterion_calc(t)

            return

        class_counts = {class_num: len(t[t == class_num]) for class_num in class_list}
        self.label = max(class_counts.items(), key=lambda num: num[1])[0]
        self.error = self._criterion_calc(t)

        num_features = x.shape[1]
        self.info_gain = 0.0

        if self.random_state is not None:
            np.random.seed(self.random_state)

        feature_loop_list = list(np.random.permutation(num_features))

        for feature in feature_loop_list:
            unique_feature = np.unique(x[:, feature])
            split_point = (unique_feature[:-1] + unique_feature[1:]) / 2.0

            for threshold in split_point:
                target_threshold_left = t[x[:, feature] <= threshold]
                target_threshold_right = t[x[:, feature] > threshold]

                val = self._info_gain_criterion(t, target_threshold_left, target_threshold_right)

                if self.info_gain < val:
                    self.info_gain = val
                    self.feature = feature
                    self.threshold = threshold

        if self.info_gain == 0.:
            return

        if depth == self.max_depth:
            return

        x_left = x[x[:, self.feature] <= self.threshold]
        t_left = t[x[:, self.feature] <= self.threshold]

        self.left_branch = _Node(self.criterion, self.max_depth)
        self.left_branch.create_threshold(x_left, t_left, depth + 1, class_list)

        x_right = x[x[:, self.feature] > self.threshold]
        t_right = t[x[:, self.feature] > self.threshold]

        self.right_branch = _Node(self.criterion, self.max_depth)
        self.right_branch.create_threshold(x_right, t_right, depth + 1, class_list)

    def predict(self, x):
        if self.feature is None or self.depth == self.max_depth:
            return self.label

        else:
            if x[self.feature] <= self.threshold:
                return self.left_branch.predict(x)
            else:
                return self.right_branch.predict(x)


class TreeFeatureImportance(object):
    def __init__(self):
        self.num_feature = None
        self.importance = None

    def _compute_feature_importance(self, node):
        if node.feature is None:
            return

        self.importance[node.feature] += node.info_gain * node.num_samples

        self._compute_feature_importance(node.left_branch)
        self._compute_feature_importance(node.right_branch)

    def get_feature_importance(self, node, num_features, normalize=True):
        self.num_feature = num_features
        self.importance = np.zeros(num_features)

        self._compute_feature_importance(node)
        self.importance /= node.num_samples

        if normalize:
            normalize = np.sum(self.importance)

            if normalize > 0.:
                self.importance /= normalize

        return self.importance


class ClassificationDecisionTree(object):
    def __init__(self, indicator="gini", max_depth=None, random_state=None):
        self.tree = None
        self.indicator = indicator
        self.max_depth = max_depth
        self.random_state = random_state

        self.tree_features = TreeFeatureImportance()

        self.feature_importance_ = None

    def fit(self, x, t):
        self.tree = _Node(self.indicator, self.max_depth, self.random_state)
        self.tree.create_threshold(x, t, 0, np.unique(t))

        self.feature_importance_ = self.tree_features.get_feature_importance(self.tree, x.shape[1])

    def predict(self, x):
        pred = []

        for sample in x:
            pred.append(self.tree.predict(sample))

        return np.array(pred)

    def score(self, x, t):
        match_list = list(np.array(self.predict(x) == t).astype("i"))
        score = sum(match_list) / float(len(t))

        return score

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


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
    def __init__(self, criterion="gini", max_depth=None, random_state=None):
        self.tree = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state

        self.tree_features = TreeFeatureImportance()

        self.feature_importance_ = None

    def fit(self, x, t):
        self.tree = _Node(self.criterion, self.max_depth, self.random_state)
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


class ClassificationRandomForest(object):
    def __init__(self, criterion="gini", n_trees: int = 10, random_state=None):
        self._n_trees = n_trees
        self.random_state = random_state
        self.criterion = criterion

        self._forest = []
        self._using_features = []

        self._classes = None
        self.feature_importance_ = None

    def _bootstrap_sampling(self, x, t):
        """
        To sample the data as bootstrap(allow restoration).
        And, sampling features axis.
        """
        n_features = x.shape[1]
        n_features_forest = np.floor(np.sqrt(n_features))
        bootstrap_x = []
        bootstrap_t = []
        np.random.seed(self.random_state)
        for i in range(self._n_trees):
            index = np.random.choice(len(t), size=len(t))
            cols = np.random.choice(n_features, size=int(n_features_forest), replace=False)
            bootstrap_x.append(x[np.ix_(index, cols)])
            bootstrap_t.append(t[index])
            self._using_features.append(cols)

        return bootstrap_x, bootstrap_t

    def fit(self, x, t):
        self._classes = np.unique(t)

        bootstrap_x, bootstrap_t = self._bootstrap_sampling(x, t)

        for i, (x_data, t_data) in enumerate(zip(bootstrap_x, bootstrap_t)):
            tree = ClassificationDecisionTree(criterion=self.criterion, random_state=self.random_state)
            tree.fit(x_data, t_data)
            self._forest.append(tree)

        self.feature_importance_ = np.zeros(x.shape[1])
        for feature, tree in zip(self._using_features, self._forest):
            self.feature_importance_[feature] += tree.feature_importance_

        self.feature_importance_ /= self._n_trees

    def predict(self, x):
        if len(self._forest) == 0:
            raise Exception("Please create forest at first.")
        predict = np.zeros((len(x), len(self._classes)))
        predict_votes = [tree.predict(x[:, using_features]) for tree, using_features in zip(self._forest, self._using_features)]
        for vote in predict_votes:
            predict[np.arange(len(vote)), vote] += 1

        predict_result = np.argmax(predict, axis=1)

        return predict_result

    def score(self, x, t):
        match_list = list(np.array(self.predict(x) == t).astype("i"))
        score = sum(match_list) / float(len(t))

        return score


if __name__ == "__main__":
    data = load_iris()
    x = data.data
    t = data.target
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)

    random_forest = ClassificationRandomForest(random_state=3)
    decision_tree = ClassificationDecisionTree(random_state=3)
    random_forest.fit(x_train, t_train)
    decision_tree.fit(x_train, t_train)
    score_decision = decision_tree.score(x_test, t_test)
    print("=" * 20, "Decision Tree", "=" * 20)
    print(score_decision)
    print(decision_tree.feature_importance_)
    score = random_forest.score(x_test, t_test)
    print("=" * 20, "Random Forest", "=" * 20)
    print(score)
    print(random_forest.feature_importance_)

import numpy as np

import marquetry
from marquetry.machine_learning import MachineLearning


class RandomForest(MachineLearning):
    """A random forest ensemble model for classification or regression tasks.

        The RandomForest class implements a random forest ensemble model, which combines multiple decision trees
        to create a high-performance classification or regression model. It is capable of training on labeled data
        and making predictions on new data points.

        Args:
            n_trees (int): The number of decision trees in the ensemble.
            target_type (str): The type of target container, either "classification" or "regression".
            max_depth (int or None): The maximum depth of individual decision trees.
                If None, the trees can grow indefinitely.
            min_split_samples (int or None): The minimum number of samples required to split a node in a tree.
                If None, defaults to 1.
            criterion (str): The criterion used for splitting nodes in the decision trees.
                Should be one of {"gini", "entropy"} for classification or {"rss", "mae"} for regression.
            seed (int or None): The random seed for reproducibility.

        Attributes:
            forest (list): The Decision Tree list to ensemble the output values.

        Examples:
            >>> model = RandomForest(n_trees=5, target_type="regression", max_depth=3, criterion="rss")
            >>> model.fit(training_data, training_labels)
            >>> predictions = model.predict(new_data)
            >>> len(model.forest)
            5

    """

    _classification_criterion = ("gini", "entropy")
    _regression_criterion = ("rss", "mae")

    def __init__(self, n_trees=10, target_type="classification", max_depth=None, min_split_samples=None,
                 criterion="gini", seed=None):
        super().__init__()
        if target_type == "classification":
            self.tree = marquetry.ml.ClassificationTree
            if criterion not in self._classification_criterion:
                raise ValueError("classification criterion should be selected from `gini` or `entropy`, but got {}"
                                 .format(criterion))

        elif target_type == "regression":
            self.tree = marquetry.ml.RegressionTree
            if criterion not in self._regression_criterion:
                raise ValueError("regression criterion should be selected from `rss` or `mae`, but got {}"
                                 .format(criterion))

        else:
            raise ValueError("target type should be `classification` or `regression`, but got {}".format(target_type))

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_split_samples = min_split_samples
        self.seed = seed
        self.criterion = criterion

        self.target_type = target_type

        self.forest = []
        self.using_feature = []

        self.unique_classes = None

    def _fit_method(self, x, t):
        if self.target_type == "classification":
            self.unique_classes = np.unique(t)
        bootstrap_x, bootstrap_t = self._bootstrap_sampling(x, t)
        for i, (x_data, t_data) in enumerate(zip(bootstrap_x, bootstrap_t)):
            tree = self.tree(self.max_depth, self.min_split_samples, self.criterion, self.seed)
            tree.fit(x_data, t_data)
            self.forest.append(tree)

    def _predict_method(self, x):
        if len(self.forest) == 0:
            raise Exception("Please create forest at first.")

        predict_vote = [
            tree.predict(x[:, using_features]) for tree, using_features in zip(self.forest, self.using_feature)]
        if self.target_type == "classification":
            predict = np.zeros((len(x), len(self.unique_classes)))
            for vote in predict_vote:
                predict[np.arange(len(vote)), vote.data] += 1

            predict_result = np.argmax(predict, axis=1)
        else:
            predict_result = None
            for vote in predict_vote:
                if predict_result is None:
                    predict_result = vote
                else:
                    predict_result += vote
            predict_result /= len(predict_vote)

        return predict_result

    def _bootstrap_sampling(self, x, t):
        n_features = x.shape[1]
        n_features_forest = np.floor(np.sqrt(n_features))

        bootstrap_x = []
        bootstrap_t = []

        np.random.seed(self.seed)
        for i in range(self.n_trees):
            index = np.random.choice(len(t), size=int(len(t)))
            features = np.random.choice(n_features, size=int(n_features_forest), replace=False)
            bootstrap_x.append(x[np.ix_(index, features)])
            bootstrap_t.append(t[index])

            self.using_feature.append(features)

        return bootstrap_x, bootstrap_t

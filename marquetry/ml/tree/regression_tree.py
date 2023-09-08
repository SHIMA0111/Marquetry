from marquetry import utils
from marquetry.ml import ClassificationTree


class RegressionTree(ClassificationTree):
    """A decision tree regressor for predicting numerical targets.

        The RegressionTree class implements a decision tree for predicting numerical targets based on input features.
        It is capable of training on labeled data with numerical targets and making predictions on new data points.

        Args:
            max_depth (int or None): The maximum depth of the decision tree.
                If None, the tree can grow indefinitely.
            min_split_samples (int or None): The minimum number of samples required to split a node.
                If None, defaults to 1.
            criterion (str): The criterion used for splitting nodes.
                Should be one of {"rss", "mae"}.
            seed (int or None): The random seed for reproducibility.

        Attributes:
            tree (dict or None): The internal representation of the decision tree.

        Methods:
            fit(self, x, t): Train the decision tree on the input data and target labels.
            predict(self, x): Make predictions on new data points.
            score(self, x, t, evaluator): Evaluate the performance of the decision tree using an evaluator function.

        Examples:
            >>> model = RegressionTree(max_depth=5, criterion="rss", seed=42)
            >>> model.fit(training_data, training_labels)
            >>> predictions = model.predict(new_data)
            >>> model.tree
            {
                'feature': 8,
                'threshold': -0.00016962857797942404,
                'content': 'branch',
                'left_branch': ...,
                'right_branch': ...,
            }

    """

    _expect_criterion = ("rss", "mae")

    def __init__(self, max_depth=None, min_split_samples=None, criterion="rss", seed=None):
        super().__init__(max_depth, min_split_samples, criterion, seed)

    def _fit_method(self, x, t):
        self.tree = self._recurrent_create_tree(x, t, 0)

    def _recurrent_create_tree(self, x, t, depth):
        is_leaf = True if len(x) < self.min_split_samples or depth == self.max_depth else False
        is_leaf, (value, impurity), feature, threshold = (
            utils.split_branch(x, t, criterion=self.criterion,
                               seed=self.seed, target_type="regression", is_leaf=is_leaf))

        if is_leaf:
            tmp_dict = {
                "content": "leaf",
                "label": value,
                "train_impurity": impurity
            }

        else:
            x_left = x[x[:, feature] <= threshold]
            t_left = t[x[:, feature] <= threshold]
            x_right = x[x[:, feature] > threshold]
            t_right = t[x[:, feature] > threshold]

            tmp_dict = {
                "feature": feature,
                "threshold": threshold,
                "content": "branch",
                "left_branch": self._recurrent_create_tree(x_left, t_left, depth + 1),
                "right_branch": self._recurrent_create_tree(x_right, t_right, depth + 1)
            }

        return tmp_dict

import numpy as np

from marquetry import utils
from marquetry import MachineLearning


class ClassificationTree(MachineLearning):
    """A decision tree classifier for classification tasks.

        The ClassificationTree class implements a decision tree for classifying data points based on their features.
        It is capable of training on labeled data and making predictions on new data points.

        Args:
            max_depth (int or None): The maximum depth of the decision tree.
                If None, the tree can grow indefinitely.
            min_split_samples (int or None): The minimum number of samples required to split a node.
                If None, defaults to 1.
            criterion (str): The criterion used for splitting nodes.
                Should be one of {"gini", "entropy"}.
            seed (int or None): The random seed for reproducibility.

        Attributes:
            tree (dict or None): The internal representation of the decision tree.

        Methods:
            fit(self, x, t): Train the decision tree on the input data and target labels.
            predict(self, x): Make predictions on new data points.
            score(self, x, t, evaluator): Evaluate the performance of the decision tree using an evaluator function.

        Examples:
            >>> model = ClassificationTree(max_depth=5, criterion="entropy", seed=42)
            >>> model.fit(training_data, training_labels)
            >>> predictions = model.predict(new_data)
            >>> model.tree
            {
                'feature': 2,
                'threshold': 2.45,
                'content': 'branch',
                'left_branch': ...,
                'right_branch':...,
            }

    """

    _expect_criterion = ("gini", "entropy")

    def __init__(self, max_depth=None, min_split_samples=None, criterion="gini", seed=None):
        if criterion not in self._expect_criterion:
            raise ValueError("criterion expects {}, but got {}".
                             format(",".join(self._expect_criterion), criterion))

        self.criterion = criterion
        self.seed = seed
        self.max_depth = max_depth if max_depth is not None else float("inf")
        self.min_split_samples = min_split_samples if min_split_samples is not None else 1

        self.tree = None
        self.unique_list = None

    def _fit_method(self, x, t):

        self.unique_list = np.unique(t).tolist()
        self.tree = self._recurrent_create_tree(x, t, 0)

    def _predict_method(self, x):

        predict_result = []

        for sample in x:
            predict_result.append(self._recurrent_prediction(sample, self.tree))

        return np.array(predict_result)

    def _recurrent_create_tree(self, x, t, depth):
        is_leaf = True if len(x) < self.min_split_samples or depth == self.max_depth else False
        is_leaf, (label, impurity), feature, threshold = (
            utils.split_branch(x, t, self.unique_list, criterion=self.criterion, seed=self.seed, is_leaf=is_leaf))

        if is_leaf:
            tmp_dict = {
                "content": "leaf",
                "label": label,
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

    def _recurrent_prediction(self, x, tree: dict):
        variable = tree["content"]

        if variable == "branch":
            threshold = tree["threshold"]
            feature = tree["feature"]
            if x[feature] <= threshold:
                return self._recurrent_prediction(x, tree["left_branch"])
            else:
                return self._recurrent_prediction(x, tree["right_branch"])
        elif variable == "leaf":
            return tree["label"]
        else:
            raise Exception("Something internal implement wrong please notify this to the developer.")

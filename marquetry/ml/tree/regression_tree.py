from marquetry import utils
from marquetry.ml import ClassificationTree


class RegressionTree(ClassificationTree):
    _expect_criterion = ("rss", "mae")

    def __init__(self, max_depth=None, min_split_samples=None, criterion="rss", seed=None):
        super().__init__(max_depth, min_split_samples, criterion, seed)

    def fit(self, x, t):
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

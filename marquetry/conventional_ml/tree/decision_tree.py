import numpy as np

import marquetry.utils as utils


class DecisionTree(object):
    def __init__(self, max_depth=None, min_split_samples=None, criterion="gini", seed=None):
        self.criterion = criterion
        self.seed = seed
        self.max_depth = max_depth if max_depth is not None else float("inf")
        self.min_split_samples = min_split_samples if min_split_samples is not None else 1

        self.tree = None
        self.unique_list = None

    def fit(self, x, t):
        self.unique_list = np.unique(t).tolist()
        self.tree = self._recurrent_create_tree(x, t, 0)

    def predict(self, x):
        predict_result = []

        for sample in x:
            predict_result.append(self._recurrent_prediction(sample, self.tree))

        return np.array(predict_result)

    def score(self, x, t):
        predict = self.predict(x)
        match_list = list(np.array(predict == t).astype("i"))
        score_data = sum(match_list) / float(len(t))

        return score_data

    def _recurrent_create_tree(self, x, t, depth, seed=None):
        is_leaf = True if len(x) < self.min_split_samples or depth == self.max_depth else False
        is_leaf, (label, impurity), feature, threshold = (
            utils.split_branch(x, t, self.unique_list, criterion=self.criterion, seed=seed, is_leaf=is_leaf))

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

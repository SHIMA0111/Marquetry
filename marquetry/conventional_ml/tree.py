import numpy as np

import marquetry
import marquetry.utils as utils
from marquetry import Model


class DecisionTree(Model):
    def __init__(self, max_depth=None, min_split_samples=None, criterion="gini", seed=None):
        super().__init__()
        self.criterion = criterion
        self.seed = seed
        self.max_depth = max_depth if max_depth is not None else float("inf")
        self.min_split_samples = min_split_samples if min_split_samples is not None else 1

        self.tree = None
        self.unique_list = None

    def forward(self, x, t=None):
        if marquetry.Config.train_mode:
            if t is None:
                raise Exception("In train mode, you need to input correct label as `t`.")
            self.unique_list = np.unique(t).tolist()
            self.tree = self._recurrent_create_tree(x, t, 0)

        pred = []
        for sample in x:
            pred.append(self._recurrent_prediction(sample, self.tree))

        return np.array(pred)

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


class RandomForest(Model):
    def __init__(self, n_trees=10, max_depth=None,
                 min_split_samples=None, criterion="gini", ensemble_method="max", seed=None):
        super().__init__()
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_split_samples = min_split_samples
        self.criterion = criterion
        self.ensemble_method = ensemble_method
        self.seed = seed

        self.forest = []
        self.using_feature = []

        self.unique_classes = None

    def forward(self, x, t=None):
        if marquetry.Config.train_mode:
            self.unique_classes = np.unique(t)
            bootstrap_x, bootstrap_t = self._bootstrap_sampling(x, t)
            predict_vote = []
            for i, (x_data, t_data) in enumerate(zip(bootstrap_x, bootstrap_t)):
                tree = DecisionTree(self.max_depth, self.min_split_samples, self.criterion, self.seed)
                predict_vote.append(tree(x_data, t_data))
                self.forest.append(tree)
        else:
            predict_vote = [tree(x[:, using_features]) for tree, using_features in zip(self.forest, self.using_feature)]

        if len(self.forest) == 0:
            raise Exception("Please create forest at first.")

        predict = np.zeros((len(x), len(self.unique_classes)))
        for vote in predict_vote:
            predict[np.arange(len(vote)), vote] += 1

        predict_result = np.argmax(predict, axis=1)

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


def score(predict, correct):
    match_list = list(np.array(predict == correct).astype("i"))
    score_data = sum(match_list) / float(len(correct))

    return score_data


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    from marquetry.datasets import Titanic

    titanic = Titanic(drop_columns=["name", "cabin", "ticket"])
    titanic_x = titanic.source
    titanic_t = titanic.target.reshape(-1)
    test_titanic = Titanic(train=False, drop_columns=["name", "cabin", "ticket"])
    test_titanic_x = test_titanic.source
    test_titanic_t = test_titanic.target.reshape(-1)

    # model = DecisionTree(min_split_samples=len(test_titanic_t) * 0.01, seed=3)
    model = RandomForest(seed=3)
    pred = model(titanic_x, titanic_t)
    train_score = score(pred, titanic_t)
    with marquetry.test_mode():
        test_pred = model(test_titanic_x)
    test_score = score(test_pred, test_titanic_t)

    print("Train_score", train_score)
    print("Test_score", test_score)

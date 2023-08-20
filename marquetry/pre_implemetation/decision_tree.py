import pprint

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split


# class DecisionTree(object):
#     tree = dict()
#
#     def fit(self, x, y, depth=4):
#         self.depth = depth
#         self.num_columns = x.shape[1]
#         self.min_values = np.amin(x, axis=0)
#         self.max_values = np.amax(x, axis=0)
#         self.tree = self.find_smallest_mean_gini(x, y, 1)
#
#
#     def gini(self, y, classes):
#         sum = 0
#         for c in classes:
#             count = np.count_nonzero(y == c)
#             sum += (count / len(y)) ** 2
#
#         return 1 - sum
#
#     def mean_gini(self, x, y, column, value):
#         classes = np.unique(y)
#         group_a_x = x[x[:, column] < value, :]
#         group_b_x = x[x[:, column] >= value, :]
#
#         if len(group_a_x) == 0 or len(group_b_x) == 0:
#             return None
#
#         group_a_y = y[x[:, column] < value]
#         group_b_y = y[x[:, column] >= value]
#
#         gini_a = self.gini(group_a_y, classes)
#         gini_b = self.gini(group_b_y, classes)
#
#         return (len(group_a_x) / len(x)) * gini_a + (len(group_b_x) / len(x)) * gini_b
#
#     def find_smallest_mean_gini(self, x, y, current_depth, prev_column=None, prev_value=None, condition=None):
#         if current_depth > self.depth:
#             counts = None
#             if condition == "lower":
#                 group_a_y = y[x[:, prev_column] < prev_value]
#                 counts = np.bincount(group_a_y)
#
#             if condition == "greater":
#                 group_b_y = y[x[:, prev_column] >= prev_value]
#                 counts = np.bincount(group_b_y)
#             label = np.argmax(counts)
#
#             return dict({
#                 "label": label,
#                 "is_edge": True
#             })
#
#         gini_scores = []
#
#         for column in range(self.num_columns):
#             if prev_column == column:
#                 continue
#
#             all_value = x[:, column]
#             min_value = self.min_values[column]
#             max_value = self.max_values[column]
#
#             for value in np.arange(min_value, max_value, 0.1):
#                 mean_gini = self.mean_gini(x, y, column, value)
#
#                 if mean_gini is not None:
#                     gini_scores.append(dict({
#                         "column": column,
#                         "value": value,
#                         "gini": mean_gini
#                     }))
#
#             sorted_scores = sorted(gini_scores, key=lambda d: d["gini"])
#             if len(sorted_scores) == 0:
#                 counts = np.bincount(y)
#                 label = np.argmax(counts)
#
#                 return dict({
#                     "label": label,
#                     "is_edge": True
#                 })
#             best_score = sorted_scores[0]
#             condition = dict({
#                 "column": best_score["column"],
#                 "value": best_score["value"],
#                 "children": dict(),
#                 "is_edge": False,
#                 "depth": current_depth
#             })
#
#             if current_depth <= self.depth:
#                 group_a_x = x[x[:, best_score["column"]] < best_score["value"], :]
#                 group_b_x = x[x[:, best_score["column"]] >= best_score["value"], :]
#
#                 if len(group_a_x) != 0:
#                     group_a_y = y[x[:, best_score["column"]] < best_score["value"]]
#                     condition["children"]["lower"] = self.find_smallest_mean_gini(
#                         group_a_x, group_a_y, current_depth + 1,
#                         best_score["column"], best_score["value"], "lower")
#
#                 if len(group_b_x) != 0:
#                     group_b_y = y[x[:, best_score["column"]] >= best_score["value"]]
#                     condition["children"]["greater"] = self.find_smallest_mean_gini(
#                         group_b_x, group_b_y, current_depth + 1,
#                         best_score["column"], best_score["value"], "greater")
#
#         return condition
#
#     def predict(self, x):
#         result = []
#         for x_data in x:
#             node = self.tree
#             for _ in range(self.depth):
#                 if node["is_edge"] is True:
#                     break
#                 if x_data[node["column"]] < node["value"]:
#                     node = node["children"]["lower"]
#                 elif x_data[node["column"]] >= node["value"]:
#                     node = node["children"]["greater"]
#
#             result.append(node["label"])
#
#         return np.array(result)
#
#     def score(self, x, t):
#         return sum(self.predict(x) == t) / float(len(t))


class _Node(object):
    def __init__(self, indicator: str = "gini", max_depth=None, random_state=None):
        self.indicator = indicator
        self.max_depth = max_depth
        self.random_state = random_state

        self.depth = None
        self.label = None
        self.error = None
        self.lift_value = None
        self.feature = None
        self.threshold = None
        self.left_branch = None
        self.right_branch = None
        self.num_data_per_class = None
        self.num_samples = None

    def _indicator_calc(self, target):
        classes = np.unique(target)
        num_data = len(target)

        if self.indicator == "gini":
            val = 1
            for class_num in classes:
                rate = float(len(target[target == class_num])) / num_data
                val -= rate ** 2

        elif self.indicator == "entropy":
            val = 0
            for class_num in classes:
                rate = float(len(target[target == class_num])) / num_data

                if rate != 0.:
                    val -= rate * np.log2(rate)

        else:
            raise Exception("You can use 'gini' or 'entropy' as indicator, but you input {}.".format(self.indicator))

        return val

    def _lift_indicator_value(self, target_origin, target_left, target_right):
        indicator_value_origin = self._indicator_calc(target_origin)
        indicator_value_left = self._indicator_calc(target_left)
        indicator_value_right = self._indicator_calc(target_right)

        mean_indicator_data = ((len(target_left) / len(target_origin) * indicator_value_left) +
                               (len(target_right) / len(target_origin) * indicator_value_right))

        lift_rate = indicator_value_origin - mean_indicator_data

        return lift_rate

    def create_threshold(self, x, t, depth, class_list):
        self.depth = depth

        self.num_samples = len(t)
        self.num_data_per_class = [len(t[t == class_num]) for class_num in class_list]

        if len(np.unique(t)) == 1:
            self.label = t[0]
            self.error = self._indicator_calc(t)

            return

        class_counts = {class_num: len(t[t == class_num]) for class_num in class_list}
        self.label = max(class_counts.items(), key=lambda x: x[1])[0]
        self.error = self._indicator_calc(t)

        num_features = x.shape[1]
        self.lift_value = 0.0

        if self.random_state is not None:
            np.random.seed(self.random_state)

        feature_loop_list = list(np.random.permutation(num_features))

        for feature in feature_loop_list:
            unique_feature = np.unique(x[:, feature])
            split_point = (unique_feature[:-1] + unique_feature[1:]) / 2.0

            for threshold in split_point:
                target_threshold_left = t[x[:, feature] <= threshold]
                target_threshold_right = t[x[:, feature] > threshold]

                val = self._lift_indicator_value(t, target_threshold_left, target_threshold_right)

                if self.lift_value < val:
                    self.lift_value = val
                    self.feature = feature
                    self.threshold = threshold

        if self.lift_value == 0.:
            return

        if depth == self.max_depth:
            return

        x_left = x[x[:, self.feature] <= self.threshold]
        t_left = t[x[:, self.feature] <= self.threshold]

        self.left_branch = _Node(self.indicator, self.max_depth, self.random_state)
        self.left_branch.create_threshold(x_left, t_left, depth + 1, class_list)

        x_right = x[x[:, self.feature] > self.threshold]
        t_right = t[x[:, self.feature] > self.threshold]

        self.right_branch = _Node(self.indicator, self.max_depth, self.random_state)
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

        self.importance[node.feature] += node.lift_value * node.num_samples

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


def main():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    max_depth = None
    random_state = 3

    clf_m = ClassificationDecisionTree(max_depth=max_depth, random_state=random_state)
    clf_m.fit(x_train, y_train)
    my_score = clf_m.score(x_test, y_test)

    clf_s = DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=random_state)
    clf_s.fit(x_train, y_train)
    sklearn_score = clf_s.score(x_test, y_test)

    print("=" * 50)
    print("Score:", str(my_score))
    print("SKLEARN Score", str(sklearn_score))

    for f_name, f_importance in zip(np.array(iris.feature_names), clf_m.feature_importance_):
        print("My:", f_name, ":", f_importance)

    for f_name, f_importance in zip(np.array(iris.feature_names), clf_s.feature_importances_):
        print("Sklearn", f_name, ":", f_importance)


if __name__ == "__main__":
    main()

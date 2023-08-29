import numpy as np

from marquetry import Model


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

    def fit(self, x, t):
        self.unique_classes = np.unique(t)
        bootstrap_x, bootstrap_t = self._bootstrap_sampling(x, t)
        for i, (x_data, t_data) in enumerate(zip(bootstrap_x, bootstrap_t)):
            tree = DecisionTree(self.max_depth, self.min_split_samples, self.criterion, self.seed)
            tree.fit(x_data, t_data)
            self.forest.append(tree)

    def predict(self, x):
        if len(self.forest) == 0:
            raise Exception("Please create forest at first.")

        predict_vote = [
            tree.predict(x[:, using_features]) for tree, using_features in zip(self.forest, self.using_feature)]

        predict = np.zeros((len(x), len(self.unique_classes)))
        for vote in predict_vote:
            predict[np.arange(len(vote)), vote] += 1

        predict_result = np.argmax(predict, axis=1)

        return predict_result

    def score(self, x, t):
        predict = self.predict(x)
        match_list = list(np.array(predict == t).astype("i"))
        score_data = sum(match_list) / float(len(t))

        return score_data

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

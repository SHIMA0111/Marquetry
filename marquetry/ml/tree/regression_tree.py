import numpy as np

import marquetry


class RegressionTree(marquetry.ml.ClassificationTree):
    _expect_criterion = ("rss", "mae")

    def __init__(self, max_depth=None, min_split_samples=None, criterion="rss", seed=None):
        super().__init__(max_depth, min_split_samples, criterion, seed)

    def _recurrent_create_tree(self, x, t, depth, seed=None):
        pass

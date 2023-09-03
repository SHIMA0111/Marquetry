import marquetry
from marquetry import cuda_backend
from marquetry import Function


class FScore(Function):
    def __init__(self, threshold):
        if not 0. <= threshold <= 1.:
            raise ValueError("threshold should be in (0.0, 1.0), but got {}".format(threshold))

        self.threshold = threshold

    def forward(self, y, t):
        precision_value = marquetry.functions.precision(y, t, self.threshold)
        recall_value = marquetry.functions.recall(y, t, self.threshold)

        self.retain_inputs(())
        if precision_value == 0. and recall_value == 0.:
            return 0.0
        else:
            f_score_value = 2 * precision_value * recall_value / (precision_value + recall_value)
            return f_score_value.data


def f_score(y, t, threshold=0.7):
    return FScore(threshold)(y, t)


class MultiFScore(Function):
    def __init__(self, target_class):
        self.target_class = target_class

    def forward(self, y, t):
        precision_value = marquetry.functions.multi_precision(y, t, self.target_class)
        recall_value = marquetry.functions.multi_recall(y, t, self.target_class)

        self.retain_inputs(())
        if precision_value == 0. and recall_value == 0.:
            return 0.0
        else:
            f_score_value = 2 * precision_value * recall_value / (precision_value + recall_value)
            return f_score_value.data


def multi_f_score(y, t, target_class=1):
    return MultiFScore(target_class)(y, t)

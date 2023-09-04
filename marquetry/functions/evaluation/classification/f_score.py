from marquetry import cuda_backend
from marquetry import Function


class FScore(Function):
    def __init__(self, threshold):
        if not 0. <= threshold <= 1.:
            raise ValueError("threshold should be in (0.0, 1.0), but got {}".format(threshold))

        self.threshold = threshold

    def forward(self, y, t):
        xp = cuda_backend.get_array_module(y)

        assert len(xp.unique(t)) <= 2

        pred = xp.asarray((y >= self.threshold), dtype="f").reshape(t.shape)

        true_positive_num = pred[t == 1].sum()
        pred_positive_num = pred.sum()
        target_positive_num = xp.asarray((t == 1), dtype="f").sum()

        precision_value, recall_value = _precision_recall_validator(true_positive_num, pred_positive_num,
                                                                    target_positive_num, xp=xp)

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
        xp = cuda_backend.get_array_module(y)

        assert len(xp.unique(t)) > 2

        pred = xp.argmax(y, axis=1).reshape(t.shape)

        pred_positive = xp.asarray(pred == self.target_class, dtype="f")
        true_map = xp.asarray(pred == t, dtype="f")

        true_positive_num = xp.asarray(pred_positive * true_map, dtype="f").sum()
        pred_positive_num = pred_positive.sum()
        target_positive_num = xp.asarray(t == self.target_class, dtype="f").sum()

        precision_value, recall_value = _precision_recall_validator(true_positive_num, pred_positive_num,
                                                                    target_positive_num, xp=xp)

        self.retain_inputs(())
        if precision_value == 0. and recall_value == 0.:
            return 0.0
        else:
            f_score_value = 2 * precision_value * recall_value / (precision_value + recall_value)
            return f_score_value.data


def multi_f_score(y, t, target_class=1):
    return MultiFScore(target_class)(y, t)


def _precision_recall_validator(true_positive_num, pred_positive_num, target_positive_num, xp):
    if pred_positive_num == 0 and target_positive_num == 0:
        precision_value = xp.asarray(0.0)
        recall_value = xp.asarray(0.0)
    elif pred_positive_num == 0:
        precision_value = xp.asarray(0.0)
        recall_value = xp.asarray(true_positive_num / target_positive_num)
    elif target_positive_num == 0:
        precision_value = xp.asarray(true_positive_num / pred_positive_num)
        recall_value = xp.asarray(0.0)
    else:
        precision_value = xp.asarray(true_positive_num / pred_positive_num)
        recall_value = xp.asarray(true_positive_num / target_positive_num)

    return precision_value, recall_value

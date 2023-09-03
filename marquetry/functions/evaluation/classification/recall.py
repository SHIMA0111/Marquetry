from marquetry import cuda_backend
from marquetry import Function


class Recall(Function):
    def __init__(self, threshold):
        if not 0. <= threshold <= 1.:
            raise ValueError("threshold should be in (0.0, 1.0), but got {}".format(threshold))
        self.threshold = threshold

    def forward(self, y, t):
        xp = cuda_backend.get_array_module(y)

        assert len(xp.unique(t)) == 2

        pred = xp.asarray((y >= self.threshold), dtype="f").reshape(t.shape)

        t_positive_num = xp.asarray((t == 1), dtype="f").sum()
        true_positive_num = pred[t == 1].sum()

        self.retain_inputs(())
        if t_positive_num == 0:
            return 0.0
        else:
            return true_positive_num / t_positive_num


def recall(y, t, threshold=0.7):
    return Recall(threshold)(y, t)


class MultiRecall(Function):
    def __init__(self, target_class):
        self.target_class = target_class

    def forward(self, y, t):
        xp = cuda_backend.get_array_module(y)

        assert len(xp.unique(t)) > 2

        pred = xp.argmax(y, axis=1).reshape(t.shape)

        t_positive = xp.asarray((t == self.target_class), dtype="f")

        pred_positive = xp.asarray((pred == self.target_class), dtype="f")
        true_positive_num = (pred_positive * t_positive).sum()
        t_positive_num = t_positive.sum()

        self.retain_inputs(())
        if t_positive_num == 0:
            return 0.0
        else:
            return true_positive_num / t_positive_num


def multi_recall(y, t, target_class=1):
    return MultiRecall(target_class)(y, t)

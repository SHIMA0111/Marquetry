from marquetry import cuda_backend
from marquetry import Function


class Precision(Function):
    def __init__(self, threshold):
        if not 0. <= threshold <= 1.:
            raise ValueError("threshold should be in (0.0, 1.0), but got {}".format(threshold))
        self.threshold = threshold

    def forward(self, y, t):
        xp = cuda_backend.get_array_module(y)

        assert len(xp.unique(t)) <= 2

        pred = xp.asarray((y >= self.threshold), dtype="f").reshape(t.shape)

        pred_positive_num = pred.sum()
        true_positive_num = pred[t == 1].sum()

        self.retain_inputs(())
        if pred_positive_num == 0:
            return xp.asarray(0.0, dtype=y.dtype)
        else:
            return xp.asarray(true_positive_num / pred_positive_num, dtype=y.dtype)


def precision(y, t, threshold=0.7):
    return Precision(threshold)(y, t)


class MultiPrecision(Function):
    def __init__(self, target_class):
        self.target_class = target_class

    def forward(self, y, t):
        xp = cuda_backend.get_array_module(y)

        assert len(xp.unique(t)) > 2

        pred = xp.argmax(y, axis=1).reshape(t.shape)

        pred_positive = xp.asarray(pred == self.target_class, dtype="f")
        pred_true_map = xp.asarray(pred == t, dtype="f")

        pred_true_positive = (pred_positive * pred_true_map).sum()
        pred_positive_num = pred_positive.sum()

        self.retain_inputs(())
        if pred_positive_num == 0:
            return xp.asarray(0.0, dtype=y.dtype)
        else:
            return xp.asarray(pred_true_positive / pred_positive_num, dtype=y.dtype)


def multi_precision(y, t, target_class=1):
    return MultiPrecision(target_class)(y, t)

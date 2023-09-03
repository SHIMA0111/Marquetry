from marquetry import cuda_backend
from marquetry import Function


class ClassificationError(Function):
    def __init__(self, ignore_label):
        self.ignore_label = ignore_label

    def forward(self, y, t):
        xp = cuda_backend.get_array_module(y)

        self.retain_inputs(())
        if self.ignore_label is not None:
            mask = xp.asarray(t == self.ignore_label).astype("f")
            ignore_cnt = mask.sum()

            pred = xp.where(mask, self.ignore_label, y.argmax(axis=1).reshape(t.shape))

            count = t.size - xp.asarray(pred == t).sum()
            total = t.size - ignore_cnt

            if total == 0:
                return xp.asarray(0.0, dtype=y.dtype)
            else:
                return xp.asarray(float(count) / total, dtype=y.dtype)

        else:
            pred = y.argmax(axis=1).reshape(t.shape)
            return xp.asarray((pred != t)).mean(dtype=y.dtype)


def classification_error(y, t, threshold=0.7):
    return ClassificationError(threshold)(y, t)


class BinaryClassificationError(Function):
    def __init__(self, threshold):
        if not 0. <= threshold <= 1.:
            raise ValueError("threshold should be in (0.0, 1.0), but got {}".format(threshold))

        self.threshold = threshold

    def forward(self, y, t):
        xp = cuda_backend.get_array_module(y)

        assert len(xp.unique(t)) > 2

        pred = xp.asarray((y >= self.threshold), dtype="f")

        self.retain_inputs(())
        return xp.asarray((pred != t), dtype="f").mean()


def binary_classification_error(y, t, threshold=0.7):
    return BinaryClassificationError(threshold)(y, t)

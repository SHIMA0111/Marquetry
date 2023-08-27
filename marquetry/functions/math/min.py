from marquetry import functions


class Min(functions.Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)

        self.retain_outputs((0,))
        return y


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)

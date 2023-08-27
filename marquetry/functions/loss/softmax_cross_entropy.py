from marquetry import Function, cuda_backend, functions, utils


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        xp = cuda_backend.get_array_module(x)
        batch_size = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[xp.arange(batch_size), t.ravel()]
        y = -log_p.sum() / xp.float32(batch_size)

        return y

    def backward(self, inputs, grad_y):
        x, t = inputs
        grad_y = grad_y[0]

        batch_size, data_dim = x.shape

        grad_y *= 1 / batch_size
        y = functions.softmax(x)
        xp = cuda_backend.get_array_module(t)
        if y.size != t.size:
            # convert class num to one-hot
            t_onehot = xp.eye(data_dim, dtype=t.dtype)[t]
        else:
            t_onehot = t

        y = (y - t_onehot) * grad_y
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

import numpy as np

from marquetry import cuda_backend
from marquetry import Layer, Parameter


class Embedding(Layer):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.w = Parameter(np.random.randn(vocab_size, embed_size))

    def __call__(self, x):
        if cuda_backend.get_array_module(x) is not np:
            self.to_gpu()

        y = self.w[x]

        return y

    def set_embedding_vector(self, vector):
        self.w.data = vector

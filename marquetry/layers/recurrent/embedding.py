import numpy as np

from marquetry import cuda_backend
from marquetry import Container
from marquetry import Layer
from marquetry import Parameter


class Embedding(Layer):
    """Embedding layer for text and categorical data.

        The Embedding layer is commonly used for encoding text data or categorical data.
        It maps discrete values (e.g., words or categories) to dense vectors,
        allowing neural networks to process such data effectively.

        Args:
            vocab_size (int): The size of the vocabulary, i.e., the number of unique words or categories.
            embed_size (int): The size of the embedding vectors.

        Attributes:
            w (marquetry.Parameter): The embedding weight matrix.

        Examples:
            >>> x = np.arange(0, 10)
            >>> x
            array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            >>> vocab_size = 10000
            >>> embed_size = 100
            >>> embedding = Embedding(vocab_size, embed_size)
            >>> y = embedding(x)
            >>> y.shape
            (10, 100)

    """
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.w = Parameter(np.random.randn(vocab_size, embed_size))
        self._fixed_embedding = False

    def __call__(self, x):
        if cuda_backend.get_array_module(x) is not np:
            self.to_gpu()

        y = self.w[x]

        return y

    def params(self):
        """Yield learnable parameters (nothing if the embedding vector is fixed)."""
        if not self._fixed_embedding:
            yield from super().params()

    def to_cpu(self):
        """Move all parameters to the CPU, including the fixed embedding vector."""
        super().to_cpu()
        if self._fixed_embedding:
            self.w.to_cpu()

    def to_gpu(self):
        """Move all parameters to the GPU, including the fixed embedding vector."""
        super().to_gpu()
        if self._fixed_embedding:
            self.w.to_gpu()

    def clear_grads(self):
        """Clear gradients of all parameters, including the fixed embedding vector."""
        super().clear_grads()
        if self._fixed_embedding:
            self.w.clear_grad()

    def set_embedding_vector(self, vector):
        """Set custom embedding vectors for the layer.

            After set the custom vector, this layer doesn't learn the parameter anymore
            because the custom vector should be pre-trained by word2vec or so.

            Args:
                vector (:class:`marquetry.Container` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                    The custom embedding vectors to set.

            Notes:
                vector:
                    If custom vector is used,
                    the custom vector shape should match the input `vector_size` and `embed_size`.
        """

        if self.w.shape != vector.shape:
            raise ValueError("embed_vector shape expected {}, but got {}"
                             .format(self.w.shape, vector.shape))

        if isinstance(vector, Container):
            vector = vector.data

        self.w.data = vector
        self._fixed_embedding = True

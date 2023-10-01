import marquetry
from marquetry import functions
from marquetry import Layer
from marquetry.layers import Linear


class RNN(Layer):
    """Recurrent Neural Network (RNN) layer for sequence modeling.

        The RNN layer is a fundamental building block for processing sequential data.
        It performs a recurrent　computation on input data, allowing it to maintain hidden states
        and capture dependencies across time steps.

        Args:
            hidden_size (int): The size of the hidden state in the RNN layer.
            in_size (int or None): The size of the input data.

        Caution:
            in_size:
                This is automatically determined from the input data shape
                and does not need to be specified except a special use case.

        Attributes:
            h (marquetry.Container or None): The current hidden state of the RNN.

        Examples:
            >>> dataset = marquetry.datasets.SinCurve()
            >>> dataloader = marquetry.dataloaders.SeqDataLoader(dataset, batch_size=32)
            >>> model = (RNN(128), marquetry.layers.Linear(1))
            >>> loss = 0
            >>> for x, t in dataloader:
            >>>     for layer in model:
            >>>         x = layer(x)
            >>>     loss += functions.mean_squared_error(x, t)
            >>> loss
            container(261.4472488881584)

    """

    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        """Reset the hidden state of the RNN."""
        self.h = None

    def set_state(self, h):
        """Set the hidden state of the RNN to the given value.

            Args:
                h (marquetry.Container):
                    The new hidden state.

        """

        if not isinstance(h, marquetry.Container):
            raise TypeError("hidden state type should be marquetry.Container, but got {}."
                            .format(type(h).__name__))

        self.h = h

    def forward(self, x):
        if self.h is None:
            new_hidden_state = functions.tanh(self.x2h(x))
        else:
            new_hidden_state = functions.tanh(self.x2h(x) + self.h2h(self.h))

        self.h = new_hidden_state

        return new_hidden_state

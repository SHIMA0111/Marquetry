import marquetry
from marquetry import functions
from marquetry import Layer
from marquetry.layers import Linear


class GRU(Layer):
    """Gated Recurrent Unit (GRU) layer for sequence modeling.

        The GRU layer is a type of recurrent neural network (RNN) layer used for modeling sequential data. It is a variation
        of the Long Short-Term Memory (LSTM) layer and is capable of learning long-term dependencies in sequential data.

        Args:
            hidden_size (int): The size of the hidden state.
            in_size (int or None): The size of the input data.

        Caution:
            in_size:
                This is automatically determined from the input data shape
                and does not need to be specified except a special use case.

        Attributes:
            h (:class:`marquetry.Container`): The current hidden state.

        Examples:
            >>> dataset = marquetry.datasets.SinCurve()
            >>> dataloader = marquetry.dataloaders.SeqDataLoader(dataset, batch_size=32)
            >>> model = (GRU(128), marquetry.layers.Linear(1))
            >>> loss = 0
            >>> for x, t in dataloader:
            >>>     for layer in model:
            >>>         x = layer(x)
            >>>     loss += functions.mean_squared_error(x, t)
            >>> loss
            container(29.910813823186242)

    """

    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.hidden_size = hidden_size

        self.x2h = Linear(hidden_size, in_size=in_size)
        self.x2r = Linear(hidden_size, in_size=in_size)
        self.x2u = Linear(hidden_size, in_size=in_size)

        self.h2h = Linear(hidden_size, in_size=hidden_size, nobias=True)
        self.h2r = Linear(hidden_size, in_size=hidden_size, nobias=True)
        self.h2u = Linear(hidden_size, in_size=hidden_size, nobias=True)

        self.h = None

    def reset_state(self):
        """Reset the hidden state."""
        self.h = None

    def set_state(self, h: marquetry.Container):
        """Set the hidden state to a custom value.

            Args:
                h (marquetry.Container): The custom hidden state.
        """
        if not isinstance(h, marquetry.Container):
            raise ValueError("custom hidden state expected marquetry.Container type, but got {}"
                             .format(type(h).__name__))
        self.h = h

    def forward(self, x):
        if self.h is None:
            new_h = functions.tanh(self.x2h(x))

        else:
            reset_gate = functions.sigmoid(self.x2r(x) + self.h2r(self.h))
            new_h = functions.tanh(self.x2h(x) + self.h2h(reset_gate * self.h))
            update_gate = functions.sigmoid(self.x2u(x) + self.h2u(self.h))

            new_h = (1 - update_gate) * new_h + update_gate * self.h

        self.h = new_h

        return new_h

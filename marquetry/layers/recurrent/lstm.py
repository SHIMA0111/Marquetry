import marquetry
from marquetry import functions
from marquetry import Layer
from marquetry.layers import Linear


class LSTM(Layer):
    """Long Short-Term Memory (LSTM) layer for sequence modeling.

        The LSTM layer is a type of recurrent neural network (RNN) layer used for modeling sequential data.
        It is designed to capture long-term dependencies in sequential data and is a variation of the standard RNN.

        Args:
            hidden_size (int): The size of the hidden state.
            in_size (int or None): The size of the input data.

        Caution:
            in_size:
                This is automatically determined from the input data shape
                and does not need to be specified except a special use case.

        Attributes:
            h (:class:`marquetry.Container`): The current hidden state.
            c (:class:`marquetry.Container`): The current cell state.

        Examples:
            >>> dataset = marquetry.datasets.SinCurve()
            >>> dataloader = marquetry.dataloaders.SeqDataLoader(dataset, batch_size=32)
            >>> model = (LSTM(128), marquetry.layers.Linear(1))
            >>> loss = 0
            >>> for x, t in dataloader:
            >>>     for layer in model:
            >>>         x = layer(x)
            >>>     loss += functions.mean_squared_error(x, t)
            >>> loss
            container(28.759597353348106)

    """

    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        self.hidden_size = hidden_size

        self.x2hs = Linear(3 * hidden_size, in_size=in_size)
        self.x2i = Linear(hidden_size, in_size=in_size)
        self.h2hs = Linear(3 * hidden_size, in_size=hidden_size, nobias=True)
        self.h2i = Linear(hidden_size, in_size=hidden_size, nobias=True)

        self.h = None
        self.c = None

    def reset_state(self):
        """Reset the hidden state and cell state."""
        self.h = None
        self.c = None

    def set_state(self, h, c=None):
        """Set the hidden state and cell state to a custom value.

            Args:
                h (marquetry.Container): The custom hidden state.
                c (marquetry.Container or None, optional): The custom cell state.

            Caution:
                Almost general use case, the cell state should NOT set custom value
                because cell state in LSTM is used only internal information connection,
                and it should be managed automatically.
                If you don't have any special reason, you should set only hidden state.

        """
        if not isinstance(h, marquetry.Container):
            raise TypeError("hidden state type should be marquetry.Container, but got {}."
                            .format(type(h).__name__))

        if c is not None and not isinstance(c, marquetry.Container):
            raise TypeError("cell state type should be (marquetry.Container or None), but got {}."
                            .format(type(c).__name__))

        self.h = h
        if c is not None:
            self.c = c

    def forward(self, x):
        if self.h is None:
            hs = functions.sigmoid(self.x2hs(x))
            input_data = functions.tanh(self.x2i(x))
        else:
            hs = functions.sigmoid(self.x2hs(x) + self.h2hs(self.h))
            input_data = functions.tanh(self.x2i(x) + self.h2i(self.h))

        forget_gate = hs[:, :self.hidden_size]
        input_gate = hs[:, self.hidden_size:2 * self.hidden_size]
        output_gate = hs[:, 2 * self.hidden_size:]

        if self.c is None:
            c_new = input_gate * input_data
        else:
            c_new = (forget_gate * self.c) + (input_gate * input_data)

        h_new = output_gate * functions.tanh(c_new)

        self.h, self.c = h_new, c_new

        return h_new

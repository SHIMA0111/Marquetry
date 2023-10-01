import marquetry
from marquetry import functions
from marquetry import Layer


class BiLSTM(Layer):
    """Bidirectional Long Short-Term Memory (BiLSTM) layer for sequence modeling.

        The BiLSTM layer is a variation of the standard LSTM layer
        that processes input data in both forward and reverse directions.
        This allows the network to capture information from both past and future context,
        resulting in richer representations.
        About LSTM, please see :class:`marquetry.layers.LSTM`.

        Args:
            hidden_size (int): The size of the hidden state in each LSTM layer.
            in_size (int or None): The size of the input data.

        Caution:
            in_size:
                This is automatically determined from the input data shape
                and does not need to be specified except a special use case.

        Attributes:
            forward_lstm (:class:`marquetry.layers.LSTM`): Forward LSTM layer.
            reverse_lstm (:class:`marquetry.layers.LSTM`): Reverse LSTM layer.

        Examples:
            >>> dataset = marquetry.datasets.SinCurve()
            >>> dataloader = marquetry.dataloaders.SeqDataLoader(dataset, batch_size=32)
            >>> model = (BiLSTM(128), marquetry.layers.Linear(1))
            >>> loss = 0
            >>> for x, t in dataloader:
            >>>     for layer in model:
            >>>         x = layer(x)
            >>>     loss += functions.mean_squared_error(x, t)
            >>> loss
            container(36.87500179632384)

    """

    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.forward_lstm = marquetry.layers.LSTM(hidden_size, in_size=in_size)
        self.reverse_lstm = marquetry.layers.LSTM(hidden_size, in_size=in_size)

    def reset_state(self):
        self.forward_lstm.reset_state()
        self.reverse_lstm.reset_state()

    def set_state(self, h, c=None):
        """Set the hidden state and cell state to a custom value.

            Args:
                h (marquetry.Container): The custom hidden state.
                c (marquetry.Container or None): The custom cell state.

            Caution:
                Almost general use case, the cell state should NOT set custom value
                because cell state in LSTM is used only internal information connection,
                and it should be managed automatically.
                If you don't have any special reason, you should set only hidden state.

        """

        self.forward_lstm.set_state(h, c)
        self.reverse_lstm.set_state(h, c)

    def forward(self, x):
        out1 = self.forward_lstm(x)
        out2 = self.reverse_lstm(x[:, ::-1])
        out2 = out2[:, ::-1]

        output = functions.concat((out1, out2), axis=-1)

        return output

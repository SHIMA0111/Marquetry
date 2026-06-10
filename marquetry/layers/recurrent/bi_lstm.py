import marquetry
from marquetry import functions
from marquetry import Layer


class BiLSTM(Layer):
    """Bidirectional Long Short-Term Memory (BiLSTM) layer for sequence modeling.

        The BiLSTM layer processes a whole sequence in both the forward and the reverse
        time direction and concatenates the hidden states of both directions per step.
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

            Unlike the step-wise :class:`marquetry.layers.LSTM`, this layer takes the
            **whole sequence at once** as a 3-dim array ``(batch, time, features)``
            because the reverse direction needs the future steps.
            The internal states are reset at the beginning of every call,
            so each call processes its sequence independently.

        Attributes:
            forward_lstm (:class:`marquetry.layers.LSTM`): Forward LSTM layer.
            reverse_lstm (:class:`marquetry.layers.LSTM`): Reverse LSTM layer.

        Examples:
            >>> x = np.random.randn(8, 20, 4).astype("f")  # (batch, time, features)
            >>> bi_lstm = BiLSTM(32)
            >>> y = bi_lstm(x)
            >>> y.shape
            (8, 20, 64)

    """

    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.forward_lstm = marquetry.layers.LSTM(hidden_size, in_size=in_size)
        self.reverse_lstm = marquetry.layers.LSTM(hidden_size, in_size=in_size)

    def reset_state(self):
        self.forward_lstm.reset_state()
        self.reverse_lstm.reset_state()

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("BiLSTM expects a sequence input shaped (batch, time, features), "
                             "but got {}-dim input.".format(x.ndim))

        self.reset_state()

        time_steps = x.shape[1]

        forward_outputs = []
        reverse_outputs = []
        for step in range(time_steps):
            forward_outputs.append(self.forward_lstm(x[:, step]))
            reverse_outputs.append(self.reverse_lstm(x[:, time_steps - 1 - step]))

        reverse_outputs.reverse()

        step_outputs = [functions.unsqueeze(functions.concat((forward_out, reverse_out), axis=1), 1)
                        for forward_out, reverse_out in zip(forward_outputs, reverse_outputs)]

        return functions.concat(step_outputs, axis=1)

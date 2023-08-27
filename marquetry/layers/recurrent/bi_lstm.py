import marquetry
import marquetry.functions as funcs
from marquetry import Layer


class BiLSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.forward_lstm = marquetry.layers.LSTM(hidden_size, in_size=in_size)
        self.reverse_lstm = marquetry.layers.LSTM(hidden_size, in_size=in_size)

    def reset_state(self):
        self.forward_lstm.reset_state()
        self.reverse_lstm.reset_state()

    def forward(self, x):
        out1 = self.forward_lstm(x)
        out2 = self.reverse_lstm(x[:, ::-1])
        out2 = out2[:, ::-1]

        output = funcs.concat((out1, out2), axis=-1)

        return output
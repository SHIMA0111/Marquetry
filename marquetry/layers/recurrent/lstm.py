from marquetry import functions
from marquetry import Layer
from marquetry.layers import Linear


class LSTM(Layer):
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
        self.h = None
        self.c = None

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

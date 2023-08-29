from marquetry import functions
from marquetry import Layer
from marquetry.layers import Linear


class GRU(Layer):
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
        self.h = None

    def set_state(self, h):
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

import marquetry.functions as funcs
from marquetry import Layer
from marquetry.layers import Linear


class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def set_state(self, h):
        self.h = h

    def forward(self, x):
        if self.h is None:
            new_hidden_state = funcs.tanh(self.x2h(x))
        else:
            new_hidden_state = funcs.tanh(self.x2h(x) + self.h2h(self.h))

        self.h = new_hidden_state

        return new_hidden_state

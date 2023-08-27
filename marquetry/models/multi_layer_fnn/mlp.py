from typing import List

import marquetry.functions as funcs
import marquetry.layers as layers
from marquetry import Model


class MLP(Model):
    def __init__(self, fnn_hidden_sizes: List[int], activation=funcs.sigmoid, is_dropout=True):
        super().__init__()
        self.activation = activation
        self.layers = []
        self.is_dropout = is_dropout

        for i, hidden_size in enumerate(fnn_hidden_sizes):
            layer = layers.Linear(hidden_size)
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            if self.is_dropout:
                x = funcs.dropout(layer(x))
            else:
                x = layer(x)
            x = self.activation(x)

        return self.layers[-1](x)
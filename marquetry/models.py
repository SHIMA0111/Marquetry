from marquetry import Layer
import marquetry.functions as funcs
import marquetry.layers as layers
from marquetry import utils


# ===========================================================================
# Model  base class
# ===========================================================================
class Model(Layer):
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)

        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


# ===========================================================================
# Sequential / MLP
# ===========================================================================
class Sequential(Model):
    def __init__(self, *layers_object):
        super().__init__()
        self.layers = []

        if len(layers_object) == 1:
            if isinstance(layers_object[0], (tuple, list)):
                layers_object = tuple(layers_object[0])

        for i, layer in enumerate(layers_object):
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class MLP(Model):
    def __init__(self, fnn_hidden_sizes: list[int], activation=funcs.sigmoid, is_dropout=True):
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

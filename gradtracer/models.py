from gradtracer import Layer
import gradtracer.functions as funcs
import gradtracer.layers as layers
from gradtracer import utils


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
        for i, layer in enumerate(layers_object):
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class MLP(Model):
    def __init__(self, fnn_hidden_sizes, out_size, activation=funcs.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        i = None
        for i, hidden_size in enumerate(fnn_hidden_sizes):
            layer = layers.Linear(hidden_size)
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

        layer = layers.Linear(out_size)
        setattr(self, "l" + str(i + 1), layer)
        self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))

        return self.layers[-1](x)

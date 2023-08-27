from marquetry import Model


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

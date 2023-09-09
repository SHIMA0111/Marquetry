from marquetry import Model


class Sequential(Model):
    """A sequential neural network model for deep learning.

        The Sequential class allows you to define a neural network model by stacking multiple layers
        sequentially. This class is commonly used for building deep neural networks where layers are
        applied in a sequential order.

        Args:
            *layers_object: Variable-length argument list of layer objects. These layers will be stacked
                in the order they are passed.

        Architecture:
            Input Layer:
                The input data.
            Sequentially Stacked Layers:
                Multiple layers, such as convolutional layers, fully connected layers,
                and activation layers, stacked in the order they are defined.
            Output Layer:
                The final layer's output.

        Examples:
            >>> model = Sequential(
            >>>     marquetry.layers.Convolution2D(32, (3, 3)),
            >>>     marquetry.functions.relu,
            >>>     marquetry.layers.MaxPooling2D((2, 2)),
            >>>     marquetry.layers.Linear(128),
            >>>     marquetry.layers.Softmax(10))
            >>> y = model(data)

    """

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

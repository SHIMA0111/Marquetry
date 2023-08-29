import marquetry.layers as layers
from marquetry import functions
from marquetry import Model


class CNN(Model):
    def __init__(self, out_size, activation=functions.relu, in_channels=None):
        super().__init__()

        self.conv1 = layers.Convolution2D(32, (3, 3), in_channels=in_channels)
        self.conv2 = layers.Convolution2D(64, (3, 3))
        self.fnn1 = layers.Linear(512)
        self.fnn2 = layers.Linear(out_size)

        self.activation = activation

    def forward(self, x):
        y = self.activation(self.conv1(x))
        y = self.activation(self.conv2(y))
        y = functions.max_pooling_2d(y, kernel_size=(2, 2), stride=2)
        y = functions.dropout(y, 0.25)
        y = functions.flatten(y)

        y = self.activation(self.fnn1(y))
        y = functions.dropout(y, 0.5)
        y = self.fnn2(y)

        return y

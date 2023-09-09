from typing import List

import marquetry.layers as layers
from marquetry import functions
from marquetry import Model


class MLP(Model):
    """A multilayer perceptron (MLP) model for various machine learning tasks.

        The MLP class implements a multilayer perceptron model,
        which is a type of feedforward neural network with multiple hidden layers.
        It can be used for various machine learning tasks, including classification and regression.

        Args:
            fnn_hidden_sizes (List[int]):
                A list of integers representing the number of hidden units in each hidden layer.
            activation (function):
                The activation function applied after each hidden layer. Default is Sigmoid function.
            is_dropout (bool): Whether to apply dropout regularization after each hidden layer.

        Architecture:
            Input Layer:
                The input features.
            Hidden Layers:
                Multiple fully connected (dense) layers with specified hidden unit sizes and activation functions.
            Output Layer:
                The final output layer.

        Example:
            >>> dataset = marquetry.datasets.MNIST()
            >>> dataloader = marquetry.dataloaders.DataLoader(dataset, batch_size=32)
            >>> model = MLP([512, 1024, 128, 10], activation=marquetry.functions.relu)
            >>> loss_data, acc_data = 0, 0
            >>> iters = 0
            >>> for x, t in dataloader:
            >>>     y = model(x)
            >>>     loss = marquetry.functions.softmax_cross_entropy(y, t)
            >>>     acc = marquetry.functions.evaluation.accuracy(y, t)
            >>>     loss_data += loss.data
            >>>     acc_data += acc.data
            >>> print("acc: {:.4f}, loss: {:.4f}".format(acc_data / iters, loss_data / iters))
            acc: 0.0997, loss: 2.3467

    """

    def __init__(self, fnn_hidden_sizes: List[int], activation=functions.sigmoid, is_dropout=True):
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
                x = functions.dropout(layer(x))
            else:
                x = layer(x)
            x = self.activation(x)

        return self.layers[-1](x)

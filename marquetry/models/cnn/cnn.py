import marquetry.layers as layers
from marquetry import functions
from marquetry import Model


class CNN(Model):
    """A convolutional neural network (CNN) model for image processing tasks.

        The CNN class implements a convolutional neural network model,
        which is widely used for image processing tasks such as image classification and object detection.
        It consists of convolutional layers, fully connected layers, and pooling layers.

        Args:
            out_size (int): The size of the output layer, typically representing the number of classes.
            activation (function): The activation function applied after each convolutional and fully connected layer.
            in_channels (int or None): The number of input channels. If None, it will be determined automatically.

        Architecture:
            Convolutional Layer 1:
                Applies 32 filters of size (3, 3) with specified activation function.
            Convolutional Layer 2:
                Applies 64 filters of size (3, 3) with specified activation function.
            Max Pooling Layer:
                Performs max pooling with a kernel size of (2, 2) and stride of 2.
            Dropout Layer:
                Applies dropout with a probability of 0.25.
            Fully Connected Layer 1:
                Has 512 units with specified activation function.
            Dropout Layer:
                Applies dropout with a probability of 0.5.
            Fully Connected Layer 2 (Output Layer):
                Produces final output with size out_size.

        Examples:
            >>> model = CNN(10, activation=marquetry.functions.relu)
            >>> data = marquetry.datasets.MNIST(transform=None)
            >>> dataloader = marquetry.dataloaders.DataLoader(data, batch_size=32)
            >>> loss_data, acc_data = 0, 0
            >>> iters = 0
            >>> for x, t in dataloader:
            >>>     iters += 1
            >>>     y = model(x)
            >>>     loss = marquetry.functions.softmax_cross_entropy(y, t)
            >>>     accuracy = marquetry.functions.evaluation.accuracy(y, t)
            >>>     loss_data += loss.data
            >>>     acc_data += accuracy.data
            >>> print("acc: {:.4f}, loss: {:.4f}".format(acc_data / iters, loss_data / iters))
            acc: 0.1099, loss: 94.3346

    """

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

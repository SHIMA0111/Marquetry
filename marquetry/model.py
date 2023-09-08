from marquetry import Layer
from marquetry import utils


# ===========================================================================
# Model  base class
# ===========================================================================
class Model(Layer):
    """Base class of all Models.

        This class inherits the Layer class.
        The deference is only add :meth:`plot` method which is to output computation graph.

        More details to see :class:`Layer`.

    """
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)

        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

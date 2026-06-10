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

    def export_onnx(self, inputs, file_path=None, **kwargs):
        """Export this model to an ONNX inference graph.

            This requires the ``onnx`` package.
            See :func:`marquetry.onnx_export.export_onnx` for the details and options.

            Args:
                inputs: A sample input array (or :class:`marquetry.Container`),
                    or a tuple/list of them for multi-input models.
                file_path (str or None): If given, the model is also serialized to this path.

            Returns:
                onnx.ModelProto: The exported model.
        """
        from marquetry.onnx_export import export_onnx

        return export_onnx(self, inputs, file_path, **kwargs)

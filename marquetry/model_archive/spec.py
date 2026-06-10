"""Format definition of the marquetry model archive.

    A marquetry archive (recommended extension: ``.mq``) is a zip file:

    .. code-block:: text

        model.mq
        ├── format.json    # format name / version metadata
        ├── graph.json     # traced computation graph (operator names + attributes)
        └── weights.npz    # parameters, captured constants and tensor attributes

    The graph stores each traced :class:`marquetry.Function` verbatim
    (class name plus the constructor attributes listed in ``ATTRIBUTE_SPEC``),
    so loading a graph re-applies the original marquetry functions
    without any semantic translation.
"""
import numpy as np


FORMAT_NAME = "marquetry_archive"
FORMAT_VERSION = 1

FORMAT_FILE = "format.json"
GRAPH_FILE = "graph.json"
WEIGHTS_FILE = "weights.npz"


class ArchiveError(RuntimeError):
    """Raised when a model can't be saved to or loaded from a marquetry archive."""


# Constructor attributes to serialize for every supported function class.
# Runtime caches (masks, saved activations, ...) are deliberately excluded:
# they are (re)computed by ``forward``.
ATTRIBUTE_SPEC = {
    # math
    "Add": (), "Sub": (), "Mul": (), "Div": (), "Neg": (),
    "Pow": ("c",),
    "Absolute": (), "Square": (),
    "Sqrt": ("eps",),
    "Reciprocal": ("dtype",),
    "Exp": (), "Log": (), "Log2": (), "Log10": (),
    "Clip": ("x_min", "x_max"),
    "MatMul": (),
    "Sum": ("axis", "keepdims"),
    "Max": ("axis", "keepdims"),
    "Min": ("axis", "keepdims"),
    "Average": ("axis", "keepdims"),
    "SumTo": ("shape",),
    "BroadcastTo": ("shape",),

    # array
    "Reshape": ("shape",),
    "Transpose": ("axes",),
    "Concat": ("axis",),
    "Squeeze": ("axis",),
    "UnSqueeze": ("axis",),
    "Split": ("indices", "axis"),
    "GetItem": ("slices",),
    "Repeat": ("repeats", "axis"),

    # connection
    "Linear": (),
    "Convolution2D": ("stride", "pad"),
    "Deconvolution2D": ("stride", "pad", "out_size"),

    # pooling
    "MaxPooling2D": ("kernel_size", "stride", "pad"),

    # normalization
    "BatchNormalization": ("avg_mean", "avg_var", "decay", "eps"),
    "LayerNormalization": ("eps",),
    "L2Normalization": ("eps", "axis"),

    # activations
    "ReLU": (), "Sigmoid": (), "Tanh": (), "Mish": (), "Identity": (),
    "LeakyReLU": ("slope",),
    "Softmax": ("axis",),
    "LogSoftmax": ("axis",),
    "Softplus": ("beta",),
    "GELU": ("approximate",),
    "GLU": ("axis",),
    "PReLU": (),
    "Swish": ("beta",),
    "DynamicSwish": (),

    # regularization
    "Dropout": ("dropout_rate",),
}


class _BatchDim(object):
    """Sentinel marking a dimension that follows the runtime batch size."""

    def __repr__(self):
        return "BATCH_DIM"


BATCH_DIM = _BatchDim()


def encode_value(value, add_tensor):
    """Encode an attribute value into a JSON-serializable structure.

        Args:
            value: The attribute value taken from a traced function.
            add_tensor (callable): Stores an ndarray and returns its npz key.

        Returns:
            A JSON-serializable representation of the value.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is BATCH_DIM:
        return {"$batch_dim": True}
    if isinstance(value, tuple):
        return {"$tuple": [encode_value(item, add_tensor) for item in value]}
    if isinstance(value, list):
        return [encode_value(item, add_tensor) for item in value]
    if isinstance(value, slice):
        return {"$slice": [encode_value(part, add_tensor)
                           for part in (value.start, value.stop, value.step)]}
    if isinstance(value, np.ndarray):
        return {"$tensor": add_tensor(value)}

    raise ArchiveError(
        "attribute value of type `{}` can't be stored in a marquetry archive."
        .format(type(value).__name__))


def decode_value(value, get_tensor):
    """Decode a structure produced by :func:`encode_value`.

        Args:
            value: The JSON-decoded structure.
            get_tensor (callable): Resolves an npz key to its ndarray.

        Returns:
            The restored attribute value.
    """
    if isinstance(value, list):
        return [decode_value(item, get_tensor) for item in value]
    if isinstance(value, dict):
        if "$tuple" in value:
            return tuple(decode_value(item, get_tensor) for item in value["$tuple"])
        if "$slice" in value:
            start, stop, step = (decode_value(part, get_tensor) for part in value["$slice"])
            return slice(start, stop, step)
        if "$tensor" in value:
            return get_tensor(value["$tensor"])
        if "$batch_dim" in value:
            return BATCH_DIM
        raise ArchiveError("unknown encoded value: {}".format(value))

    return value


def contains_batch_dim(value):
    """Check whether a decoded attribute value contains the batch sentinel."""
    if value is BATCH_DIM:
        return True
    if isinstance(value, slice):
        return any(contains_batch_dim(part)
                   for part in (value.start, value.stop, value.step))
    if isinstance(value, (tuple, list)):
        return any(contains_batch_dim(item) for item in value)

    return False


def resolve_batch_dim(value, batch_size):
    """Replace every batch sentinel in a decoded attribute value with ``batch_size``."""
    if value is BATCH_DIM:
        return batch_size
    if isinstance(value, slice):
        return slice(resolve_batch_dim(value.start, batch_size),
                     resolve_batch_dim(value.stop, batch_size),
                     resolve_batch_dim(value.step, batch_size))
    if isinstance(value, tuple):
        return tuple(resolve_batch_dim(item, batch_size) for item in value)
    if isinstance(value, list):
        return [resolve_batch_dim(item, batch_size) for item in value]

    return value

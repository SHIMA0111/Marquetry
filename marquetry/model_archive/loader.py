import io
import json
import zipfile

import numpy as np

import marquetry
from marquetry.container import Parameter
from marquetry.model import Model
from marquetry.model_archive import spec
from marquetry.model_archive.spec import ArchiveError


def load_archive(file_path):
    """Load a marquetry archive saved by :func:`save_archive`.

        Args:
            file_path (str): Path to the archive file.

        Returns:
            GraphModel: A runnable model which replays the stored graph.
                Its parameters are trainable :class:`marquetry.Parameter` objects,
                so the loaded model supports both inference and further training.
    """
    try:
        with zipfile.ZipFile(file_path, "r") as archive:
            header = json.loads(archive.read(spec.FORMAT_FILE))
            graph = json.loads(archive.read(spec.GRAPH_FILE))
            weights_buffer = io.BytesIO(archive.read(spec.WEIGHTS_FILE))
    except (zipfile.BadZipFile, KeyError) as error:
        raise ArchiveError(
            "`{}` is not a marquetry archive: {}".format(file_path, error))

    if header.get("format") != spec.FORMAT_NAME:
        raise ArchiveError(
            "`{}` is not a marquetry archive (format: {}).".format(
                file_path, header.get("format")))
    if header.get("version") != spec.FORMAT_VERSION:
        raise ArchiveError(
            "archive version {} isn't supported by this marquetry "
            "(supported version: {}).".format(header.get("version"), spec.FORMAT_VERSION))

    npz_file = np.load(weights_buffer)
    weights = {key: npz_file[key] for key in npz_file.files}

    return GraphModel(graph, weights, name=header.get("model_name"))


class GraphModel(Model):
    """A model reconstructed from a marquetry archive.

        The stored graph is replayed by re-applying the original
        :mod:`marquetry.functions` classes node by node, so the loaded model
        behaves like the traced one: backpropagation, train/test mode dispatch
        (dropout, batch normalization) and parameter updates all work.
    """

    def __init__(self, graph, weights, name=None):
        super().__init__()

        self._graph_name = name or "GraphModel"
        self._input_names = [info["name"] for info in graph["inputs"]]
        self._output_names = list(graph["outputs"])

        self._parameter_names = []
        for param_name, npz_key in graph["parameters"].items():
            setattr(self, param_name, Parameter(weights[npz_key], name=param_name))
            self._parameter_names.append(param_name)

        self._constants = {const_name: weights[npz_key]
                           for const_name, npz_key in graph["constants"].items()}

        self._nodes = []
        for node in graph["nodes"]:
            function_class = _resolve_function_class(node["op"])
            attributes = {attr_name: spec.decode_value(value, weights.__getitem__)
                          for attr_name, value in node["attrs"].items()}
            needs_batch = any(spec.contains_batch_dim(value)
                              for value in attributes.values())
            self._nodes.append((function_class, attributes, needs_batch,
                                node["inputs"], node["outputs"]))

    def forward(self, *inputs):
        if len(inputs) != len(self._input_names):
            raise ValueError(
                "the model expects {} inputs, but {} were given."
                .format(len(self._input_names), len(inputs)))

        env = dict(zip(self._input_names, inputs))
        for param_name in self._parameter_names:
            env[param_name] = getattr(self, param_name)
        env.update(self._constants)

        for function_class, attributes, needs_batch, in_names, out_names in self._nodes:
            # Each call gets a fresh function instance (required for autodiff graph
            # recording). The attribute dict is shallow-copied: scalars detach while
            # ndarray state such as batch norm running statistics stays shared so
            # train-mode updates persist across calls.
            instance_attributes = dict(attributes)
            if needs_batch:
                batch_size = _leading_dim(env[in_names[0]])
                instance_attributes = {
                    attr_name: spec.resolve_batch_dim(value, batch_size)
                    for attr_name, value in instance_attributes.items()}

            function = function_class.__new__(function_class)
            function.__dict__.update(instance_attributes)

            arguments = [env[in_name] if in_name else None for in_name in in_names]
            results = function(*arguments)
            if not isinstance(results, (tuple, list)):
                results = (results,)
            for out_name, result in zip(out_names, results):
                env[out_name] = result

        outputs = tuple(env[out_name] for out_name in self._output_names)

        return outputs if len(outputs) > 1 else outputs[0]


def _resolve_function_class(op_name):
    if op_name not in spec.ATTRIBUTE_SPEC:
        raise ArchiveError(
            "operator `{}` isn't supported by this marquetry version.".format(op_name))

    function_class = getattr(marquetry.functions, op_name, None)
    if function_class is None:
        raise ArchiveError(
            "operator `{}` can't be resolved in marquetry.functions.".format(op_name))

    return function_class


def _leading_dim(value):
    data = value.data if isinstance(value, marquetry.Container) else value

    return data.shape[0]

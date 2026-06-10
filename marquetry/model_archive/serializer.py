import io
import json
import zipfile

import numpy as np

from marquetry import cuda_backend
from marquetry import graph_tracer
from marquetry.model_archive import spec
from marquetry.model_archive.spec import ArchiveError


class _Namer(object):
    """SSA name table mapping ContainerNode identities to unique tensor names."""

    def __init__(self):
        self._names = {}
        self._taken = set()
        self._counts = {}
        # id() keys are only stable while the objects live, so hold strong refs.
        self._keepalive = []

    def fresh(self, hint):
        hint = hint or "tensor"
        count = self._counts.get(hint, 0)
        while True:
            name = hint if count == 0 else "{}_{}".format(hint, count)
            count += 1
            if name not in self._taken:
                break
        self._counts[hint] = count
        self._taken.add(name)

        return name

    def has(self, container_node):
        return id(container_node) in self._names

    def name_of(self, container_node):
        return self._names[id(container_node)]

    def assign(self, container_node, hint):
        name = self.fresh(hint)
        self._names[id(container_node)] = name
        self._keepalive.append(container_node)

        return name

    def set(self, container_node, name):
        if name in self._taken:
            raise ArchiveError("tensor name `{}` is specified twice.".format(name))
        self._taken.add(name)
        self._names[id(container_node)] = name
        self._keepalive.append(container_node)


def save_archive(model, inputs, file_path, *, dynamic_batch=True):
    """Save a model as a marquetry archive (graph + weights, ``.mq``).

        Runs one forward pass with the sample inputs to record the computation
        graph, then stores every traced function verbatim (class name and
        constructor attributes) together with the parameters, so the archive can
        be loaded and executed without the original model class.

        Args:
            model (marquetry.Model or marquetry.Layer): The model to save.
            inputs: A sample input array (or :class:`marquetry.Container`), or a
                tuple/list of them for multi-input models.
            file_path (str): Destination path. ``.mq`` is the recommended extension.
            dynamic_batch (bool): If True, reshape targets whose leading dim follows
                the batch axis are stored as batch-relative so the loaded model
                accepts any batch size.
    """
    input_containers = graph_tracer.normalize_sample_inputs(inputs)
    outputs = graph_tracer.trace_forward(model, input_containers)
    parameter_lookup = graph_tracer.build_parameter_lookup(model)

    functions = graph_tracer.topological_functions(outputs)

    namer = _Namer()
    arrays = {}
    parameters = {}
    constants = {}

    first_input = input_containers[0]
    batch_size = first_input.shape[0] if first_input.ndim >= 1 else None

    input_names = ["input"] if len(input_containers) == 1 else [
        "input_{}".format(index) for index in range(len(input_containers))]
    output_names = ["output"] if len(outputs) == 1 else [
        "output_{}".format(index) for index in range(len(outputs))]

    for container, name in zip(input_containers, input_names):
        namer.set(container.node, name)

    output_aliases = []
    for container, name in zip(outputs, output_names):
        if namer.has(container.node):
            output_aliases.append((namer.name_of(container.node), name))
        else:
            namer.set(container.node, name)

    if not functions and not output_aliases:
        raise ArchiveError(
            "no computation graph is recorded from the model outputs. "
            "The save needs to run under back-propagation enabled mode.")

    nodes = []
    for function in functions:
        in_names = [_resolve_input(node, namer, parameter_lookup, arrays,
                                   parameters, constants)
                    for node in function.inputs]

        out_names = []
        for reference in function.outputs:
            out_node = reference()
            if out_node is None:
                out_names.append(namer.fresh("unused"))
            elif namer.has(out_node):
                out_names.append(namer.name_of(out_node))
            else:
                out_names.append(namer.assign(out_node, function.name.lower() + "_out"))

        nodes.append(_serialize_function(function, in_names, out_names,
                                         arrays, dynamic_batch, batch_size))

    for source_name, alias_name in output_aliases:
        namer._taken.add(alias_name)
        nodes.append({"op": "Identity", "attrs": {},
                      "inputs": [source_name], "outputs": [alias_name]})

    graph = {
        "inputs": [_input_info(name, container, dynamic_batch, batch_size)
                   for name, container in zip(input_names, input_containers)],
        "outputs": output_names,
        "parameters": parameters,
        "constants": constants,
        "nodes": nodes,
    }

    header = {
        "format": spec.FORMAT_NAME,
        "version": spec.FORMAT_VERSION,
        "model_name": type(model).__name__,
    }

    weights_buffer = io.BytesIO()
    np.savez_compressed(weights_buffer, **arrays)

    with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(spec.FORMAT_FILE, json.dumps(header, indent=2))
        archive.writestr(spec.GRAPH_FILE, json.dumps(graph, indent=2))
        archive.writestr(spec.WEIGHTS_FILE, weights_buffer.getvalue())


def _resolve_input(container_node, namer, parameter_lookup, arrays,
                   parameters, constants):
    if namer.has(container_node):
        return namer.name_of(container_node)

    if container_node.creator is not None:
        raise ArchiveError(
            "internal error: encountered an input produced by an unvisited function.")

    parameter_entry = parameter_lookup.get(id(container_node))
    if parameter_entry is not None:
        key, data = parameter_entry
        name = namer.assign(container_node, key)
        arrays[name] = np.ascontiguousarray(cuda_backend.as_numpy(data))
        parameters[name] = name
        return name

    if container_node.data is None:
        # Absent optional input such as an omitted bias.
        return ""

    name = namer.assign(container_node, "const")
    arrays[name] = np.ascontiguousarray(cuda_backend.as_numpy(container_node.data))
    constants[name] = name

    return name


def _serialize_function(function, in_names, out_names, arrays,
                        dynamic_batch, batch_size):
    op_name = function.__class__.__name__
    attribute_names = spec.ATTRIBUTE_SPEC.get(op_name)
    if attribute_names is None:
        raise ArchiveError(
            "`{}` can't be stored in a marquetry archive. "
            "Please replace it or save without this computation.".format(op_name))

    attributes = {name: getattr(function, name) for name in attribute_names}

    if op_name == "Reshape":
        attributes["shape"] = _mark_batch_dim(
            attributes["shape"], function.x_shape, dynamic_batch, batch_size)

    tensor_counter = [len(arrays)]

    def add_tensor(array):
        key = "__tensor__/{}".format(tensor_counter[0])
        tensor_counter[0] += 1
        arrays[key] = np.ascontiguousarray(cuda_backend.as_numpy(array))
        return key

    encoded = {name: spec.encode_value(value, add_tensor)
               for name, value in attributes.items()}

    return {"op": op_name, "attrs": encoded,
            "inputs": in_names, "outputs": out_names}


def _mark_batch_dim(shape, x_shape, dynamic_batch, batch_size):
    """Mark a reshape target's leading dim as batch-relative when it tracks the batch.

        Mirrors the ONNX exporter heuristic: substitute only when both the traced
        input and the target keep the batch extent in front (e.g. flatten's
        ``(batch, -1)``).
    """
    if not dynamic_batch or batch_size is None:
        return shape
    if not isinstance(shape, (tuple, list)) or not shape:
        return shape
    if shape[0] != batch_size:
        return shape
    if x_shape is None or len(x_shape) < 1 or x_shape[0] != batch_size:
        return shape

    return (spec.BATCH_DIM,) + tuple(shape[1:])


def _input_info(name, container, dynamic_batch, batch_size):
    data = cuda_backend.as_numpy(container.data)
    info = {
        "name": name,
        "dtype": str(data.dtype),
        "shape": [int(dim) for dim in data.shape],
    }
    if dynamic_batch and data.ndim >= 1 and batch_size is not None \
            and data.shape[0] == batch_size:
        info["batch_axis"] = 0

    return info

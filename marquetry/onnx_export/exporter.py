import numpy as np

import onnx
from onnx import helper
from onnx import numpy_helper

import marquetry
from marquetry import cuda_backend
from marquetry import graph_tracer
from marquetry.onnx_export import converters
from marquetry.onnx_export.converters import ONNXExportError


DEFAULT_OPSET_VERSION = 21
MINIMUM_OPSET_VERSION = 18
TARGET_IR_VERSION = 10
BATCH_DIM_PARAM = "batch_size"


class ExportContext(object):
    """Mutable state shared with the op converters while a graph is being built.

        Collects the ONNX nodes and initializers, and owns the SSA name table
        which maps :class:`marquetry.container.ContainerNode` objects to unique
        tensor names.
    """

    def __init__(self, opset_version, dynamic_batch, batch_size):
        self.opset_version = opset_version
        self.dynamic_batch = dynamic_batch
        self.batch_size = batch_size

        self.nodes = []
        self.initializers = []

        self._names = {}
        self._taken = set()
        self._counts = {}
        # id() keys are only stable while the objects live, so hold strong refs.
        self._keepalive = []

    def fresh_name(self, hint):
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

    def has_name(self, container_node):
        return id(container_node) in self._names

    def name_of(self, container_node):
        return self._names[id(container_node)]

    def assign_name(self, container_node, hint):
        name = self.fresh_name(hint)
        self._names[id(container_node)] = name
        self._keepalive.append(container_node)

        return name

    def set_name(self, container_node, name):
        if name in self._taken:
            raise ONNXExportError("tensor name `{}` is specified twice.".format(name))
        self._taken.add(name)
        self._names[id(container_node)] = name
        self._keepalive.append(container_node)

    def intermediate(self, hint):
        return self.fresh_name(hint)

    def add_initializer(self, name, array):
        array = np.ascontiguousarray(cuda_backend.as_numpy(array))
        self.initializers.append(numpy_helper.from_array(array, name))

    def constant(self, array, hint, dtype=None):
        array = np.asarray(cuda_backend.as_numpy(array))
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        name = self.fresh_name(hint)
        self.add_initializer(name, array)

        return name

    def add_node(self, op_type, inputs, outputs, **attributes):
        node = helper.make_node(op_type, list(inputs), list(outputs),
                                name=self.fresh_name(op_type), **attributes)
        self.nodes.append(node)


def export_onnx(model, inputs, file_path=None, *, opset_version=DEFAULT_OPSET_VERSION,
                dynamic_batch=True, input_names=None, output_names=None, check=True):
    """Export a model to an ONNX inference graph.

        Runs one forward pass with the sample inputs in test mode, walks the recorded
        computation graph, and converts every traced function to ONNX operators.
        The produced model is pinned to ``opset_version`` and IR version 10 so that
        it stays loadable by widely deployed ONNX Runtime versions.

        Args:
            model (marquetry.Model or marquetry.Layer): The model to export.
            inputs: A sample input array (or :class:`marquetry.Container`), or a
                tuple/list of them for multi-input models. The traced graph is
                specialized to these shapes (except the batch axis).
            file_path (str or None): If given, the model is also serialized to this path.
            opset_version (int): Target opset of the default ONNX domain. Defaults to 21.
            dynamic_batch (bool): If True, the leading axis of every graph input/output
                is exported as a symbolic ``batch_size`` dimension.
            input_names (list of str or None): Names for the graph inputs.
            output_names (list of str or None): Names for the graph outputs.
            check (bool): If True, run ``onnx.checker.check_model`` on the result.

        Returns:
            onnx.ModelProto: The exported model.
    """

    if opset_version < MINIMUM_OPSET_VERSION:
        raise ValueError(
            "opset_version should be {} or later, but got {}."
            .format(MINIMUM_OPSET_VERSION, opset_version))
    if opset_version > onnx.defs.onnx_opset_version():
        raise ValueError(
            "opset_version {} isn't supported by the installed onnx package (max: {})."
            .format(opset_version, onnx.defs.onnx_opset_version()))

    input_containers = graph_tracer.normalize_sample_inputs(inputs)
    outputs = graph_tracer.trace_forward(model, input_containers)

    first_input = input_containers[0]
    batch_size = first_input.shape[0] if first_input.ndim >= 1 else None
    context = ExportContext(opset_version, dynamic_batch, batch_size)

    input_names = _default_io_names(input_names, len(input_containers), "input")
    output_names = _default_io_names(output_names, len(outputs), "output")

    for container, name in zip(input_containers, input_names):
        context.set_name(container.node, name)

    # An output that is an input itself (or appears twice) needs an Identity bridge
    # because its tensor name is already taken.
    output_aliases = []
    for container, name in zip(outputs, output_names):
        if context.has_name(container.node):
            output_aliases.append((context.name_of(container.node), name))
        else:
            context.set_name(container.node, name)

    parameter_lookup = graph_tracer.build_parameter_lookup(model)

    functions = graph_tracer.topological_functions(outputs)
    if not functions and not output_aliases:
        raise ONNXExportError(
            "no computation graph was recorded from the model outputs. "
            "The outputs must be produced by marquetry functions applied to the sample inputs.")

    for function in functions:
        in_names = [_resolve_input_name(node, context, parameter_lookup)
                    for node in function.inputs]

        out_names = []
        for reference in function.outputs:
            out_node = reference()
            if out_node is None:
                out_names.append(context.fresh_name("unused"))
            elif context.has_name(out_node):
                out_names.append(context.name_of(out_node))
            else:
                out_names.append(context.assign_name(out_node, function.name.lower() + "_out"))

        converter = converters.lookup(function)
        if converter is None:
            raise ONNXExportError(
                "`{}` can't be converted to ONNX operators. "
                "Please replace it or export without this computation."
                .format(function.name))

        converter(function, in_names, out_names, context)

    for source_name, alias_name in output_aliases:
        context._taken.add(alias_name)
        context.add_node("Identity", [source_name], [alias_name])

    graph_inputs = [_tensor_value_info(name, container.data, context)
                    for name, container in zip(input_names, input_containers)]
    graph_outputs = [_tensor_value_info(name, container.data, context)
                     for name, container in zip(output_names, outputs)]

    graph = helper.make_graph(context.nodes, type(model).__name__,
                              graph_inputs, graph_outputs,
                              initializer=context.initializers)

    model_proto = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset_version)],
        producer_name="marquetry")
    # helper.make_model stamps the installed package's latest IR version, which old
    # runtimes reject, so pin it down explicitly.
    model_proto.ir_version = TARGET_IR_VERSION

    if check:
        onnx.checker.check_model(model_proto)

    if file_path is not None:
        onnx.save(model_proto, file_path)

    return model_proto


def _default_io_names(names, count, prefix):
    if names is None:
        if count == 1:
            return [prefix]
        return ["{}_{}".format(prefix, index) for index in range(count)]

    names = list(names)
    if len(names) != count:
        raise ValueError(
            "{} names are needed for {} tensors, but got {}.".format(prefix, count, len(names)))

    return names


def _resolve_input_name(container_node, context, parameter_lookup):
    if context.has_name(container_node):
        return context.name_of(container_node)

    if container_node.creator is not None:
        raise ONNXExportError(
            "internal error: encountered an input produced by an unconverted function.")

    parameter_entry = parameter_lookup.get(id(container_node))
    if parameter_entry is not None:
        key, data = parameter_entry
        name = context.assign_name(container_node, key)
        context.add_initializer(name, data)
        return name

    if container_node.data is None:
        # Absent optional input such as a omitted bias.
        return ""

    name = context.assign_name(container_node, "const")
    context.add_initializer(name, container_node.data)

    return name


def _tensor_value_info(name, array, context):
    array = cuda_backend.as_numpy(array)
    element_type = helper.np_dtype_to_tensor_dtype(np.dtype(array.dtype))

    shape = list(array.shape)
    if (context.dynamic_batch and shape
            and context.batch_size is not None and shape[0] == context.batch_size):
        shape[0] = BATCH_DIM_PARAM

    return helper.make_tensor_value_info(name, element_type, shape)

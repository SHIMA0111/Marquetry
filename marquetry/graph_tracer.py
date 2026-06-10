"""Format-agnostic helpers to trace a model's recorded computation graph.

    The exporters (ONNX, marquetry archive) share the same first step:
    run one forward pass with sample inputs while the graph is recorded,
    then walk the creator chain backwards from the outputs.
    This module hosts that machinery without depending on any export format.
"""
import marquetry


def normalize_sample_inputs(inputs):
    """Convert user-provided sample inputs into a list of data-holding containers.

        Args:
            inputs: A sample input array (or :class:`marquetry.Container`), or a
                tuple/list of them for multi-input models.

        Returns:
            list of marquetry.Container: The validated input containers.
    """
    input_list = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
    if not input_list:
        raise ValueError("at least one sample input is needed to trace the model.")

    input_containers = []
    for sample in input_list:
        container = marquetry.as_container(sample)
        if container.data is None:
            raise ValueError("sample inputs should hold data, but got an empty container.")
        input_containers.append(container)

    return input_containers


def trace_forward(model, input_containers):
    """Run one forward pass recording the full graph in inference behavior.

        The pass runs in test mode (inference behavior for dropout, batch norm, ...)
        with back-propagation recording enabled, and every function input retained
        so that constants stay readable from the traced graph.

        Returns:
            tuple of marquetry.Container: The model outputs.
    """
    with marquetry.using_config("train", False):
        with marquetry.using_config("enable_backprop", True):
            with marquetry.using_config("retain_graph_inputs", True):
                outputs = model(*input_containers)

    return tuple(outputs) if isinstance(outputs, (tuple, list)) else (outputs,)


def topological_functions(outputs):
    """Collect every function reachable from the outputs in executable order.

        Returns:
            list of marquetry.Function: Traced functions sorted topologically.
    """
    functions = []
    seen = set()

    stack = [output.creator for output in outputs if output.creator is not None]
    while stack:
        function = stack.pop()
        if function in seen:
            continue
        seen.add(function)
        functions.append(function)

        for in_node in function.inputs:
            if in_node.creator is not None:
                stack.append(in_node.creator)

    # The generation number strictly increases along every edge,
    # so sorting by it yields a valid topological (executable) order.
    functions.sort(key=lambda recorded: recorded.generation)

    return functions


def build_parameter_lookup(model):
    """Map the node identity of every model parameter to its hierarchical name.

        Returns:
            dict: ``id(parameter.node)`` -> ``(name, data)`` for parameters with data.
    """
    lookup = {}
    if not hasattr(model, "_flatten_params"):
        return lookup

    params_dict = {}
    model._flatten_params(params_dict)
    for key, param in params_dict.items():
        if param is None or param.data is None:
            continue
        lookup[id(param.node)] = (key, param.data)

    return lookup

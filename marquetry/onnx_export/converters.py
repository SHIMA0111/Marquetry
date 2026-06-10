import numpy as np
from onnx.helper import np_dtype_to_tensor_dtype

from marquetry import utils


class ONNXExportError(RuntimeError):
    """Raised when a model contains a computation which can't be expressed in ONNX."""


_CONVERTERS = {}


def register(name):
    """Register a converter for a :class:`marquetry.Function` subclass by its class name."""
    def decorator(converter):
        _CONVERTERS[name] = converter
        return converter

    return decorator


def lookup(function_obj):
    """Get the converter for a traced function instance, or None if unsupported."""
    return _CONVERTERS.get(function_obj.__class__.__name__)


def _input_dtype(func, index=0, default=np.float32):
    dtype = func.inputs[index].dtype
    return dtype if dtype is not None else np.dtype(default)


def _cast_like_first_input(func, source_name, hint, ctx):
    """Insert a Cast to the dtype of the function's first input when dtypes differ."""
    x_dtype = func.inputs[0].dtype
    other_dtype = func.inputs[1].dtype
    if x_dtype is None or other_dtype is None or x_dtype == other_dtype:
        return source_name

    casted = ctx.intermediate(hint)
    ctx.add_node("Cast", [source_name], [casted],
                 to=np_dtype_to_tensor_dtype(np.dtype(x_dtype)))

    return casted


def _register_direct(marquetry_name, onnx_op):
    def converter(func, in_names, out_names, ctx):
        ctx.add_node(onnx_op, in_names, out_names)

    _CONVERTERS[marquetry_name] = converter


# ===========================================================================
# math
# ===========================================================================
for _marquetry_name, _onnx_op in [
    ("Add", "Add"), ("Sub", "Sub"), ("Mul", "Mul"), ("Div", "Div"),
    ("Neg", "Neg"), ("Exp", "Exp"), ("Log", "Log"), ("Absolute", "Abs"),
    ("Sqrt", "Sqrt"), ("Reciprocal", "Reciprocal"), ("MatMul", "MatMul"),
]:
    _register_direct(_marquetry_name, _onnx_op)


@register("Pow")
def _convert_pow(func, in_names, out_names, ctx):
    exponent = ctx.constant(np.asarray(func.c, dtype=_input_dtype(func)), "pow_exponent")
    ctx.add_node("Pow", [in_names[0], exponent], out_names)


@register("Square")
def _convert_square(func, in_names, out_names, ctx):
    ctx.add_node("Mul", [in_names[0], in_names[0]], out_names)


def _convert_log_base(base):
    def converter(func, in_names, out_names, ctx):
        natural_log = ctx.intermediate("natural_log")
        ctx.add_node("Log", in_names, [natural_log])
        denominator = ctx.constant(np.asarray(np.log(base), dtype=_input_dtype(func)), "log_base")
        ctx.add_node("Div", [natural_log, denominator], out_names)

    return converter


_CONVERTERS["Log2"] = _convert_log_base(2.0)
_CONVERTERS["Log10"] = _convert_log_base(10.0)


@register("Clip")
def _convert_clip(func, in_names, out_names, ctx):
    dtype = _input_dtype(func)
    inputs = [in_names[0]]
    inputs.append(ctx.constant(np.asarray(func.x_min, dtype=dtype), "clip_min")
                  if func.x_min is not None else "")
    inputs.append(ctx.constant(np.asarray(func.x_max, dtype=dtype), "clip_max")
                  if func.x_max is not None else "")
    while inputs and inputs[-1] == "":
        inputs.pop()
    ctx.add_node("Clip", inputs, out_names)


def _convert_reduce(onnx_op):
    def converter(func, in_names, out_names, ctx):
        axis = func.axis
        inputs = [in_names[0]]
        if axis is not None:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            inputs.append(ctx.constant(np.asarray(axes, dtype=np.int64), "reduce_axes"))
        ctx.add_node(onnx_op, inputs, out_names, keepdims=int(func.keepdims))

    return converter


_CONVERTERS["Sum"] = _convert_reduce("ReduceSum")
_CONVERTERS["Max"] = _convert_reduce("ReduceMax")
_CONVERTERS["Min"] = _convert_reduce("ReduceMin")
_CONVERTERS["Average"] = _convert_reduce("ReduceMean")


@register("SumTo")
def _convert_sum_to(func, in_names, out_names, ctx):
    target_shape = tuple(func.shape)
    x_shape = func.x_shape
    if x_shape is None or tuple(x_shape) == target_shape:
        ctx.add_node("Identity", in_names, out_names)
        return

    lead = len(x_shape) - len(target_shape)
    axes = list(range(lead))
    axes += [k + lead for k, dim in enumerate(target_shape) if dim == 1 and x_shape[k + lead] != 1]

    reduced = ctx.intermediate("sum_to_reduced")
    axes_name = ctx.constant(np.asarray(axes, dtype=np.int64), "sum_to_axes")
    ctx.add_node("ReduceSum", [in_names[0], axes_name], [reduced], keepdims=1)
    shape_name = ctx.constant(np.asarray(target_shape, dtype=np.int64), "sum_to_shape")
    ctx.add_node("Reshape", [reduced, shape_name], out_names)


@register("BroadcastTo")
def _convert_broadcast_to(func, in_names, out_names, ctx):
    if func.x_shape is None:
        ctx.add_node("Identity", in_names, out_names)
        return

    shape_name = ctx.constant(np.asarray(func.shape, dtype=np.int64), "expand_shape")
    ctx.add_node("Expand", [in_names[0], shape_name], out_names)


# ===========================================================================
# array
# ===========================================================================
@register("Reshape")
def _convert_reshape(func, in_names, out_names, ctx):
    shape = func.shape
    if isinstance(shape, (int, np.integer)):
        shape = (shape,)
    shape = [int(dim) for dim in shape]

    # Keep the batch axis dynamic: substitute the leading dim with 0 ("copy the
    # input dim") only when both the traced input and the target keep the batch
    # extent in front, e.g. flatten's `(batch, -1)`. A reshape whose leading dim
    # merely coincides with the batch size stays literal. Tracing with a batch
    # size that doesn't collide with fixed reshape dims avoids the remaining
    # ambiguity entirely.
    x_shape = func.x_shape
    if (ctx.dynamic_batch and ctx.batch_size is not None
            and shape and shape[0] == ctx.batch_size
            and x_shape is not None and len(x_shape) >= 1 and x_shape[0] == ctx.batch_size):
        shape[0] = 0

    shape_name = ctx.constant(np.asarray(shape, dtype=np.int64), "reshape_shape")
    ctx.add_node("Reshape", [in_names[0], shape_name], out_names)


@register("Transpose")
def _convert_transpose(func, in_names, out_names, ctx):
    if func.axes is None:
        ctx.add_node("Transpose", in_names, out_names)
    else:
        ctx.add_node("Transpose", in_names, out_names, perm=[int(axis) for axis in func.axes])


@register("Concat")
def _convert_concat(func, in_names, out_names, ctx):
    ctx.add_node("Concat", in_names, out_names, axis=int(func.axis))


@register("Squeeze")
def _convert_squeeze(func, in_names, out_names, ctx):
    inputs = [in_names[0]]
    if func.axis is not None:
        inputs.append(ctx.constant(np.asarray(func.axis, dtype=np.int64), "squeeze_axes"))
    ctx.add_node("Squeeze", inputs, out_names)


@register("UnSqueeze")
def _convert_unsqueeze(func, in_names, out_names, ctx):
    axes_name = ctx.constant(np.asarray(sorted(func.axis), dtype=np.int64), "unsqueeze_axes")
    ctx.add_node("Unsqueeze", [in_names[0], axes_name], out_names)


@register("Split")
def _convert_split(func, in_names, out_names, ctx):
    if isinstance(func.indices, (int, np.integer)):
        ctx.add_node("Split", [in_names[0]], out_names,
                     axis=int(func.axis), num_outputs=int(func.indices))
        return

    x_shape = func.inputs[0].shape
    if x_shape is None:
        raise ONNXExportError(
            "Split with explicit indices requires the input shape, which was not recorded.")

    total = x_shape[func.axis]
    indices = [int(index) for index in func.indices]
    sizes = [indices[0]]
    sizes += [second - first for first, second in zip(indices, indices[1:])]
    sizes.append(total - indices[-1])

    sizes_name = ctx.constant(np.asarray(sizes, dtype=np.int64), "split_sizes")
    ctx.add_node("Split", [in_names[0], sizes_name], out_names, axis=int(func.axis))


@register("GetItem")
def _convert_get_item(func, in_names, out_names, ctx):
    int64_info = np.iinfo(np.int64)
    slices = func.slices if isinstance(func.slices, tuple) else (func.slices,)

    if len(slices) == 1 and isinstance(slices[0], (np.ndarray, list)):
        indices = np.asarray(slices[0])
        if indices.dtype.kind not in "iu":
            raise ONNXExportError("GetItem with non-integer array indices is not supported.")
        indices_name = ctx.constant(indices.astype(np.int64), "gather_indices")
        ctx.add_node("Gather", [in_names[0], indices_name], out_names, axis=0)
        return

    starts, ends, axes, steps, squeeze_axes = [], [], [], [], []
    for axis, item in enumerate(slices):
        if isinstance(item, (int, np.integer)):
            index = int(item)
            starts.append(index)
            ends.append(index + 1 if index != -1 else int64_info.max)
            axes.append(axis)
            steps.append(1)
            squeeze_axes.append(axis)
        elif isinstance(item, slice):
            step = item.step if item.step is not None else 1
            if step > 0:
                start = item.start if item.start is not None else 0
                end = item.stop if item.stop is not None else int64_info.max
            else:
                start = item.start if item.start is not None else -1
                end = item.stop if item.stop is not None else int64_info.min
            starts.append(int(start))
            ends.append(int(end))
            axes.append(axis)
            steps.append(int(step))
        else:
            raise ONNXExportError(
                "GetItem with `{}` index is not supported.".format(type(item).__name__))

    if not axes:
        ctx.add_node("Identity", in_names, out_names)
        return

    slice_inputs = [
        in_names[0],
        ctx.constant(np.asarray(starts, dtype=np.int64), "slice_starts"),
        ctx.constant(np.asarray(ends, dtype=np.int64), "slice_ends"),
        ctx.constant(np.asarray(axes, dtype=np.int64), "slice_axes"),
        ctx.constant(np.asarray(steps, dtype=np.int64), "slice_steps"),
    ]

    if squeeze_axes:
        sliced = ctx.intermediate("sliced")
        ctx.add_node("Slice", slice_inputs, [sliced])
        squeeze_name = ctx.constant(np.asarray(squeeze_axes, dtype=np.int64), "squeeze_axes")
        ctx.add_node("Squeeze", [sliced, squeeze_name], out_names)
    else:
        ctx.add_node("Slice", slice_inputs, out_names)


# ===========================================================================
# connection
# ===========================================================================
@register("Linear")
def _convert_linear(func, in_names, out_names, ctx):
    has_bias = bool(in_names[2])
    if func.inputs[0].ndim == 2:
        inputs = in_names[:3] if has_bias else in_names[:2]
        ctx.add_node("Gemm", inputs, out_names)
    elif has_bias:
        product = ctx.intermediate("linear_matmul")
        ctx.add_node("MatMul", in_names[:2], [product])
        ctx.add_node("Add", [product, in_names[2]], out_names)
    else:
        ctx.add_node("MatMul", in_names[:2], out_names)


@register("Convolution2D")
def _convert_convolution_2d(func, in_names, out_names, ctx):
    pad_height, pad_width = func.pad
    inputs = in_names[:3] if in_names[2] else in_names[:2]
    ctx.add_node("Conv", inputs, out_names,
                 strides=list(func.stride),
                 pads=[pad_height, pad_width, pad_height, pad_width])


@register("Deconvolution2D")
def _convert_deconvolution_2d(func, in_names, out_names, ctx):
    pad_height, pad_width = func.pad
    if func.out_size is not None:
        x_shape = func.inputs[0].shape
        w_shape = func.inputs[1].shape
        expected = tuple(
            utils.get_deconvolution_outsize(size, kernel, stride, pad)
            for size, kernel, stride, pad in zip(
                x_shape[2:], w_shape[2:], func.stride, func.pad))
        if tuple(utils.pair(func.out_size)) != expected:
            raise ONNXExportError(
                "Deconvolution2D with a custom out_size is not supported in the ONNX export.")

    inputs = in_names[:3] if in_names[2] else in_names[:2]
    ctx.add_node("ConvTranspose", inputs, out_names,
                 strides=list(func.stride),
                 pads=[pad_height, pad_width, pad_height, pad_width])


# ===========================================================================
# pooling
# ===========================================================================
@register("MaxPooling2D")
def _convert_max_pooling_2d(func, in_names, out_names, ctx):
    kernel = list(utils.pair(func.kernel_size))
    strides = list(utils.pair(func.stride))
    pad_height, pad_width = utils.pair(func.pad)

    ctx.add_node("MaxPool", in_names, out_names, kernel_shape=kernel, strides=strides,
                 pads=[pad_height, pad_width, pad_height, pad_width])


# ===========================================================================
# normalization
# ===========================================================================
@register("BatchNormalization")
def _convert_batch_normalization(func, in_names, out_names, ctx):
    mean_name = ctx.constant(func.avg_mean, "batch_norm_mean")
    var_name = ctx.constant(func.avg_var, "batch_norm_var")
    ctx.add_node("BatchNormalization",
                 [in_names[0], in_names[1], in_names[2], mean_name, var_name],
                 out_names, epsilon=float(func.eps))


@register("LayerNormalization")
def _convert_layer_normalization(func, in_names, out_names, ctx):
    x_node = func.inputs[0]
    if x_node.ndim == 2:
        ctx.add_node("LayerNormalization", in_names, out_names,
                     axis=-1, epsilon=float(func.eps))
        return

    # Marquetry normalizes 4D inputs over all the non-batch axes at once,
    # so flatten, normalize the last axis, then restore the original shape.
    flat_shape = ctx.constant(np.asarray([0, -1], dtype=np.int64), "layer_norm_flat_shape")
    flattened = ctx.intermediate("layer_norm_flat")
    ctx.add_node("Reshape", [in_names[0], flat_shape], [flattened])

    normalized = ctx.intermediate("layer_norm_out")
    ctx.add_node("LayerNormalization", [flattened, in_names[1], in_names[2]], [normalized],
                 axis=-1, epsilon=float(func.eps))

    original_shape = [0] + [int(dim) for dim in x_node.shape[1:]]
    shape_name = ctx.constant(np.asarray(original_shape, dtype=np.int64), "layer_norm_shape")
    ctx.add_node("Reshape", [normalized, shape_name], out_names)


@register("L2Normalization")
def _convert_l2_normalization(func, in_names, out_names, ctx):
    dtype = _input_dtype(func)
    axis = func.axis
    axes = (axis,) if isinstance(axis, int) else tuple(axis)

    squared = ctx.intermediate("l2_norm_squared")
    ctx.add_node("Mul", [in_names[0], in_names[0]], [squared])

    axes_name = ctx.constant(np.asarray(axes, dtype=np.int64), "l2_norm_axes")
    summed = ctx.intermediate("l2_norm_sum")
    ctx.add_node("ReduceSum", [squared, axes_name], [summed], keepdims=1)

    root = ctx.intermediate("l2_norm_root")
    ctx.add_node("Sqrt", [summed], [root])

    eps_name = ctx.constant(np.asarray(func.eps, dtype=dtype), "l2_norm_eps")
    denominator = ctx.intermediate("l2_norm_denominator")
    ctx.add_node("Add", [root, eps_name], [denominator])

    ctx.add_node("Div", [in_names[0], denominator], out_names)


# ===========================================================================
# activations
# ===========================================================================
for _marquetry_name, _onnx_op in [
    ("ReLU", "Relu"), ("Sigmoid", "Sigmoid"), ("Tanh", "Tanh"),
    ("Mish", "Mish"), ("Identity", "Identity"),
]:
    _register_direct(_marquetry_name, _onnx_op)


@register("LeakyReLU")
def _convert_leaky_relu(func, in_names, out_names, ctx):
    ctx.add_node("LeakyRelu", in_names, out_names, alpha=float(func.slope))


@register("Softmax")
def _convert_softmax(func, in_names, out_names, ctx):
    ctx.add_node("Softmax", in_names, out_names, axis=int(func.axis))


@register("LogSoftmax")
def _convert_log_softmax(func, in_names, out_names, ctx):
    ctx.add_node("LogSoftmax", in_names, out_names, axis=int(func.axis))


@register("Softplus")
def _convert_softplus(func, in_names, out_names, ctx):
    if func.beta == 1:
        ctx.add_node("Softplus", in_names, out_names)
        return

    beta_name = ctx.constant(np.asarray(func.beta, dtype=_input_dtype(func)), "softplus_beta")
    scaled = ctx.intermediate("softplus_scaled")
    ctx.add_node("Mul", [in_names[0], beta_name], [scaled])
    activated = ctx.intermediate("softplus_activated")
    ctx.add_node("Softplus", [scaled], [activated])
    ctx.add_node("Div", [activated, beta_name], out_names)


@register("GELU")
def _convert_gelu(func, in_names, out_names, ctx):
    # The Gelu operator only exists since opset 20; decompose below that.
    if func.approximate in ("none", "tanh") and ctx.opset_version >= 20:
        ctx.add_node("Gelu", in_names, out_names, approximate=func.approximate)
    elif func.approximate == "none":
        _emit_gelu_erf(func, in_names, out_names, ctx)
    elif func.approximate == "tanh":
        _emit_gelu_tanh(func, in_names, out_names, ctx)
    else:
        # The sigmoid approximation has no ONNX counterpart: x * sigmoid(1.702 * x)
        scale_name = ctx.constant(np.asarray(1.702, dtype=_input_dtype(func)), "gelu_scale")
        scaled = ctx.intermediate("gelu_scaled")
        ctx.add_node("Mul", [in_names[0], scale_name], [scaled])
        gate = ctx.intermediate("gelu_gate")
        ctx.add_node("Sigmoid", [scaled], [gate])
        ctx.add_node("Mul", [in_names[0], gate], out_names)


def _emit_gelu_erf(func, in_names, out_names, ctx):
    """Exact GELU as primitives: 0.5 * x * (1 + erf(x / sqrt(2)))."""
    dtype = _input_dtype(func)

    root_two = ctx.constant(np.asarray(np.sqrt(2.0), dtype=dtype), "gelu_root_two")
    erf_input = ctx.intermediate("gelu_erf_input")
    ctx.add_node("Div", [in_names[0], root_two], [erf_input])

    erf = ctx.intermediate("gelu_erf")
    ctx.add_node("Erf", [erf_input], [erf])

    one = ctx.constant(np.asarray(1.0, dtype=dtype), "gelu_one")
    gate = ctx.intermediate("gelu_gate")
    ctx.add_node("Add", [erf, one], [gate])

    half = ctx.constant(np.asarray(0.5, dtype=dtype), "gelu_half")
    half_x = ctx.intermediate("gelu_half_x")
    ctx.add_node("Mul", [in_names[0], half], [half_x])

    ctx.add_node("Mul", [half_x, gate], out_names)


def _emit_gelu_tanh(func, in_names, out_names, ctx):
    """Tanh-approximated GELU as primitives:
        0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """
    dtype = _input_dtype(func)

    three = ctx.constant(np.asarray(3.0, dtype=dtype), "gelu_three")
    cubed = ctx.intermediate("gelu_cubed")
    ctx.add_node("Pow", [in_names[0], three], [cubed])

    cubic_coeff = ctx.constant(np.asarray(0.044715, dtype=dtype), "gelu_cubic_coeff")
    scaled_cube = ctx.intermediate("gelu_scaled_cube")
    ctx.add_node("Mul", [cubed, cubic_coeff], [scaled_cube])

    inner_sum = ctx.intermediate("gelu_inner_sum")
    ctx.add_node("Add", [in_names[0], scaled_cube], [inner_sum])

    coeff = ctx.constant(np.asarray(np.sqrt(2.0 / np.pi), dtype=dtype), "gelu_coeff")
    inner = ctx.intermediate("gelu_inner")
    ctx.add_node("Mul", [inner_sum, coeff], [inner])

    tanh = ctx.intermediate("gelu_tanh")
    ctx.add_node("Tanh", [inner], [tanh])

    one = ctx.constant(np.asarray(1.0, dtype=dtype), "gelu_one")
    gate = ctx.intermediate("gelu_gate")
    ctx.add_node("Add", [tanh, one], [gate])

    half = ctx.constant(np.asarray(0.5, dtype=dtype), "gelu_half")
    half_x = ctx.intermediate("gelu_half_x")
    ctx.add_node("Mul", [in_names[0], half], [half_x])

    ctx.add_node("Mul", [half_x, gate], out_names)


@register("GLU")
def _convert_glu(func, in_names, out_names, ctx):
    first_half = ctx.intermediate("glu_value")
    second_half = ctx.intermediate("glu_gate_source")
    ctx.add_node("Split", [in_names[0]], [first_half, second_half],
                 axis=int(func.axis), num_outputs=2)
    gate = ctx.intermediate("glu_gate")
    ctx.add_node("Sigmoid", [second_half], [gate])
    ctx.add_node("Mul", [first_half, gate], out_names)


@register("PReLU")
def _convert_prelu(func, in_names, out_names, ctx):
    # ONNX PRelu broadcasts the slope unidirectionally, so reshape the per-channel
    # alpha to the channel-aligned shape recorded in the forward pass (e.g. (1, C, 1, 1)).
    slope_source = _cast_like_first_input(func, in_names[1], "prelu_slope_cast", ctx)
    slope_shape = ctx.constant(
        np.asarray(tuple(func.ex_alpha_shape), dtype=np.int64), "prelu_slope_shape")
    slope = ctx.intermediate("prelu_slope")
    ctx.add_node("Reshape", [slope_source, slope_shape], [slope])
    ctx.add_node("PRelu", [in_names[0], slope], out_names)


def _emit_swish(in_name, out_names, beta_name, ctx):
    if beta_name is None:
        scaled = in_name
    else:
        scaled = ctx.intermediate("swish_scaled")
        ctx.add_node("Mul", [in_name, beta_name], [scaled])
    gate = ctx.intermediate("swish_gate")
    ctx.add_node("Sigmoid", [scaled], [gate])
    ctx.add_node("Mul", [in_name, gate], out_names)


@register("Swish")
def _convert_swish(func, in_names, out_names, ctx):
    if func.beta == 1:
        beta_name = None
    else:
        beta_name = ctx.constant(np.asarray(func.beta, dtype=_input_dtype(func)), "swish_beta")
    _emit_swish(in_names[0], out_names, beta_name, ctx)


@register("DynamicSwish")
def _convert_dynamic_swish(func, in_names, out_names, ctx):
    beta_name = _cast_like_first_input(func, in_names[1], "swish_beta_cast", ctx)
    _emit_swish(in_names[0], out_names, beta_name, ctx)


# ===========================================================================
# regularization
# ===========================================================================
@register("Dropout")
def _convert_dropout(func, in_names, out_names, ctx):
    # Inference graphs don't need dropout; it traces as the identity in test mode.
    ctx.add_node("Identity", in_names, out_names)

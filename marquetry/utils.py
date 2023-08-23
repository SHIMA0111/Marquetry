import os
import subprocess
import urllib.request

import numpy as np
from PIL import Image

import marquetry
from marquetry import as_variable
from marquetry import Variable
from marquetry import cuda_backend


# ===========================================================================
# Visualize for computational graph
# ===========================================================================
def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)

    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        ret += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        ret += dot_edge.format(id(f), id(y()))

    return ret


def get_dot_graph(output, verbose=True):
    """Generates a graphviz DOT text of a computational graph.

    Build a graph of functions and variables backward-reachable from the output.
    To visualize a graphviz DOT text, you need the dot binary from the graph viz
    package (www.graphviz.org).
    """
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + "}"


def plot_dot_graph(output, verbose=True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser("~"), ".module_tmp")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    img = Image.open(to_file)
    img.show()


# ===========================================================================
# utility functions for numpy calculation
# ===========================================================================
def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape."""
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(grad_y, x_shape, axis, keepdims):
    """Reshape gradient appropriately for sum's backward."""
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(grad_y.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = grad_y.shape

    grad_y = grad_y.reshape(shape)
    return grad_y


def logsumexp(x, axis=1):
    xp = cuda_backend.get_array_module(x)
    x_max = x.max(axis=axis, keepdims=True)
    y = x - x_max
    xp.exp(y, out=y)
    sum_exp = y.sum(axis=axis, keepdims=True)
    xp.log(sum_exp, out=sum_exp)
    y = x_max + sum_exp

    return y


def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]

    return shape


# ===========================================================================
# Random Forest utilities
# ===========================================================================
def class_impurity_criterion(target, criterion="gini"):
    xp = cuda_backend.get_array_module(target)
    classes = xp.unique(target)
    num_samples = len(target)

    if criterion == "gini":
        result = 1.
        for class_num in classes:
            # calc each class rate
            rate = float(len(target[target == class_num])) / num_samples
            result -= rate ** 2

    elif criterion == "entropy":
        result = 0.
        for class_num in classes:
            # calc each class rate
            rate = float(len(target[target == class_num])) / num_samples
            result -= rate * xp.log2(rate)
    else:
        raise Exception("{} is not supported as criterion.".format(criterion))

    return result


def class_information_gain(target, target_left, target_right, criterion="gini"):
    """
    information_gain indicates how much cleansing the impurity from before splitting to after.
    """
    impurity_target = class_impurity_criterion(target, criterion=criterion)
    impurity_left = class_impurity_criterion(target_left, criterion=criterion)
    impurity_right = class_impurity_criterion(target_right, criterion=criterion)

    split_mean_impurity = (float(len(target_left) / len(target)) * impurity_left +
                           float(len(target_right) / len(target) * impurity_right))
    info_gain = impurity_target - split_mean_impurity

    return info_gain


def split_branch(data, target, class_list, criterion="gini", seed=None, is_leaf=False):
    """
    return: is_leave, (label, impurity), feature, threshold
    """
    xp = cuda_backend.get_array_module(data)

    count_classes_datas = [len(target[target == class_num]) for class_num in class_list]

    current_impurity = class_impurity_criterion(target, criterion=criterion)
    class_counts = dict(zip(class_list, count_classes_datas))
    label = max(class_counts.items(), key=lambda count: count[1])[0]

    if len(xp.unique(target)) == 1:
        # If target labels already have only 1 label, the impurity is 0 and, the data can't split anymore.
        return True, (label, current_impurity), None, None

    class_counts = dict(zip(class_list, count_classes_datas))
    label = max(class_counts.items(), key=lambda count: count[1])[0]

    if is_leaf:
        return True, (label, current_impurity), None, None

    num_features = data.shape[1]
    pre_info_gain = 0.0

    xp.random.seed(seed)

    shuffle_features_list = list(xp.random.permutation(num_features))

    feature_candidate, threshold_candidate = None, None
    for feature in shuffle_features_list:
        unique_in_feature = xp.unique(data[:, feature])
        threshold_point = (unique_in_feature[:-1] + unique_in_feature[1:]) / 2.

        for threshold in threshold_point:
            target_left = target[data[:, feature] <= threshold]
            target_right = target[data[:, feature] > threshold]

            info_gain = class_information_gain(target, target_left, target_right, criterion=criterion)

            if pre_info_gain < info_gain:
                pre_info_gain = info_gain
                feature_candidate = feature
                threshold_candidate = threshold

    if pre_info_gain == 0.:
        return True, (label, current_impurity), None, None

    return False, (label, current_impurity), feature_candidate, threshold_candidate


# ===========================================================================
# Download utilities
# ===========================================================================
def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    percent = downloaded / total_size * 100
    indicator_num = int(downloaded / total_size * 30)

    percent = percent if percent < 100. else 100.
    indicator_num = indicator_num if indicator_num < 30 else 30

    indicator = "#" * indicator_num + "." * (30 - indicator_num)
    print(bar_template.format(indicator, percent), end="")


def get_file(url, file_name=None):
    if file_name is None:
        file_name = url[url.rfind("/") + 1:]

    file_path = os.path.join(marquetry.Config.CACHE_DIR, file_name)

    if not os.path.exists(marquetry.Config.CACHE_DIR):
        os.mkdir(marquetry.Config.CACHE_DIR)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)

    try:
        urllib.request.urlretrieve(url, file_path, show_progress)

    except (Exception, KeyboardInterrupt):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    print(" Done")

    return file_path


# ===========================================================================
# Gradient check
# ===========================================================================
def gradient_check(f, x, *args, rtol=1e-4, atol=1e-5, **kwargs):
    x = as_variable(x)
    xp = cuda_backend.get_array_module(x)
    x.data = x.data.astype(xp.float64)

    num_grad = numerical_grad(f, x, *args, **kwargs)
    y = f(x, *args, **kwargs)
    y.backward()
    bp_grad = x.grad.data

    assert bp_grad.shape == num_grad.shape
    res = array_close(num_grad, bp_grad, rtol=rtol, atol=atol)

    grad_diff = xp.abs(bp_grad - num_grad).sum()
    if not res:
        print("")
        print("========== FAILED (Gradient Check) ==========")
        print("Back propagation for {} failed.".format(f.__class__.__name__))
        print("Grad Diff: {}".format(grad_diff))
        print("=============================================")
    else:
        print("")
        print("========== OK (Gradient Check) ==========")
        print("Grad Diff: {}".format(grad_diff))
        print("=============================================")

    return res


def numerical_grad(func, x, *args, **kwargs):
    eps = 1e-4

    x = x.data if isinstance(x, Variable) else x
    xp = cuda_backend.get_array_module(x)
    if xp is not np:
        np_x = cuda_backend.as_numpy(x)
    else:
        np_x = x

    grad = xp.zeros_like(x)

    iters = np.nditer(np_x, flags=["multi_index"], op_flags=["readwrite"])
    while not iters.finished:
        index = iters.multi_index
        tmp_val = x[index].copy()

        x[index] = tmp_val + eps
        y1 = func(x, *args, **kwargs)
        if isinstance(y1, Variable):
            y1 = y1.data

        y1 = y1.copy()

        x[index] = tmp_val - eps
        y2 = func(x, *args, **kwargs)
        if isinstance(y2, Variable):
            y2 = y2.data

        y2 = y2.copy()

        if isinstance(y1, list):
            diff = 0
            for i in range(len(y1)):
                diff += (y1[i] - y2[i]).sum()
        else:
            diff = (y1 - y2).sum()

        if isinstance(diff, Variable):
            diff = diff.data

        grad[index] = diff / (2 * eps)

        x[index] = tmp_val
        iters.iternext()

    return grad


def array_equal(a, b):
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b

    a, b = cuda_backend.as_numpy(a), cuda_backend.as_numpy(b)

    return np.array_equal(a, b)


def array_close(a, b, rtol=1e-4, atol=1e-5):
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b

    a, b = cuda_backend.as_numpy(a), cuda_backend.as_numpy(b)

    return np.allclose(a, b, rtol, atol)


# ===========================================================================
# im2col / col2im
# ===========================================================================
def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    batch_size, channels, height, weight = img.shape
    kernel_height, kernel_width = pair(kernel_size)
    stride_height, stride_width = pair(stride)
    padding_height, padding_width = pair(pad)

    out_height = get_conv_outsize(height, kernel_height, stride_height, padding_height)
    out_width = get_conv_outsize(weight, kernel_width, stride_width, padding_width)

    xp = cuda_backend.get_array_module(img)
    img = xp.pad(img, (
        (0, 0), (0, 0),
        (padding_height, padding_height + stride_height - 1),
        (padding_width, padding_width + stride_width - 1)), mode="constant", constant_values=(0,))
    col = xp.ndarray((batch_size, channels, kernel_height, kernel_width, out_height, out_width), dtype=img.dtype)

    for height in range(kernel_height):
        height_lim = height + out_height * stride_height

        for width in range(kernel_width):
            width_lim = width + out_width * stride_width

            col[:, :, height, width, :, :] = img[:, :, height:height_lim:stride_height, width:width_lim:stride_width]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((batch_size * out_height * out_width, -1))

    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    batch_size, channels, height, width = img_shape
    kernel_height, kernel_width = pair(kernel_size)
    stride_height, stride_width = pair(stride)
    padding_height, padding_width = pair(pad)

    out_height = get_conv_outsize(height, kernel_height, stride_height, padding_height)
    out_width = get_conv_outsize(width, kernel_width, stride_width, padding_width)

    if to_matrix:
        col = (col.reshape(batch_size, out_height, out_width, channels, kernel_height, kernel_width).
               transpose(0, 3, 4, 5, 1, 2))

    xp = cuda_backend.get_array_module(col)
    img = xp.zeros(
        (
            batch_size,
            channels,
            height + 2 * padding_height + stride_height - 1,
            width + 2 * padding_width + stride_width - 1), dtype=col.dtype)

    for height_range in range(kernel_height):
        height_lim = height_range + stride_height * out_height

        for width_range in range(kernel_width):
            width_lim = width_range + stride_width * out_width

            img[:, :, height_range:height_lim:stride_height, width_range:width_lim:stride_width] += col[:, :, height_range, width_range, :, :]

    return img[:, :, padding_height:height + padding_height, padding_width:width + padding_width]


# ===========================================================================
# others
# ===========================================================================
def get_deconv_outsize(size, kernel_size, stride_size, padding_size):
    return stride_size * (size - 1) + kernel_size - 2 * padding_size


def get_conv_outsize(size, kernel_size, stride_size, padding_size):
    return (size + 2 * padding_size - kernel_size) // stride_size + 1


def pair(x):
    if isinstance(x, int):
        return x, x
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError("pair can't use {}".format(x))

import marquetry
from marquetry import cuda_backend


def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    batch_size, channels, height, weight = img.shape
    kernel_height, kernel_width = marquetry.utils.pair(kernel_size)
    stride_height, stride_width = marquetry.utils.pair(stride)
    padding_height, padding_width = marquetry.utils.pair(pad)

    out_height = marquetry.utils.get_convolution_outsize(height, kernel_height, stride_height, padding_height)
    out_width = marquetry.utils.get_convolution_outsize(weight, kernel_width, stride_width, padding_width)

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

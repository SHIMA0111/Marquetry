import marquetry
from marquetry import cuda_backend


def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    """The util of im2col main process"""

    batch_size, channels, height, weight = img.shape
    kernel_height, kernel_width = marquetry.utils.pair(kernel_size)
    stride_height, stride_width = marquetry.utils.pair(stride)
    padding_height, padding_width = marquetry.utils.pair(pad)

    # calc the output_size(output_height, output_width)
    out_height = marquetry.utils.get_convolution_outsize(height, kernel_height, stride_height, padding_height)
    out_width = marquetry.utils.get_convolution_outsize(weight, kernel_width, stride_width, padding_width)

    xp = cuda_backend.get_array_module(img)
    # Padding addition
    img = xp.pad(img, (
        (0, 0), (0, 0),
        (padding_height, padding_height + stride_height - 1),
        (padding_width, padding_width + stride_width - 1)), mode="constant", constant_values=(0,))

    # output array which has 6-dims to handle the local area extraction.
    col = xp.ndarray((batch_size, channels, kernel_height, kernel_width, out_height, out_width), dtype=img.dtype)

    for height in range(kernel_height):
        # The limit of the height first point(out_height means the kernel route point num respect stride_height.)
        # output_height * stride_height means the height point relative the origin first point.
        # (origin_point + relative_points)
        height_lim = height + out_height * stride_height

        for width in range(kernel_width):
            # the same mechanism as height_limit
            width_lim = width + out_width * stride_width

            col[:, :, height, width, :, :] = img[:, :, height:height_lim:stride_height, width:width_lim:stride_width]

    if to_matrix:
        # To more easy use, the 6-dims array convert to matrix having
        # (batch_size * output_height * output_width, channels * kernel_height * kernel_width)
        # convolution can be interpreted as a special case of the linear transformation.
        # The kernel can be changed the matrix which is (channels * kernel_height * kernel_width, out_channels)
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((batch_size * out_height * out_width, -1))

    return col

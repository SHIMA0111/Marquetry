import marquetry
from marquetry import cuda_backend


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    """The util of col2im main process"""

    batch_size, channels, height, width = img_shape
    kernel_height, kernel_width = marquetry.utils.pair(kernel_size)
    stride_height, stride_width = marquetry.utils.pair(stride)
    padding_height, padding_width = marquetry.utils.pair(pad)

    out_height = marquetry.utils.get_convolution_outsize(height, kernel_height, stride_height, padding_height)
    out_width = marquetry.utils.get_convolution_outsize(width, kernel_width, stride_width, padding_width)

    if to_matrix:
        # Reconvert to 6-dims array from the easy used matrix to convolut.
        col = (col.reshape(batch_size, out_height, out_width, channels, kernel_height, kernel_width).
               transpose(0, 3, 4, 5, 1, 2))

    xp = cuda_backend.get_array_module(col)
    # prepare the clear image array which is 4-dims.
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
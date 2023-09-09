def get_convolution_outsize(size, kernel_size, stride_size, padding_size):
    """Get convolution outsize num which is used for im2col process."""
    return (size + 2 * padding_size - kernel_size) // stride_size + 1


def get_deconvolution_outsize(size, kernel_size, stride_size, padding_size):
    """Get deconvolution outsize num which is used for col2im process."""
    return stride_size * (size - 1) + kernel_size - 2 * padding_size

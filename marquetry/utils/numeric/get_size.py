def get_deconvolution_outsize(size, kernel_size, stride_size, padding_size):
    return stride_size * (size - 1) + kernel_size - 2 * padding_size


def get_convolution_outsize(size, kernel_size, stride_size, padding_size):
    return (size + 2 * padding_size - kernel_size) // stride_size + 1
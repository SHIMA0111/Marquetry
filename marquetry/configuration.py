import contextlib
import os
import sys

from marquetry import cuda_backend


class Config(object):
    """Configuration of the marquetry.

        This class implements the configuration of the marquetry dynamic computation graph.
        When a user use this class if user want to adjust themselves use case.

        Args:
            enable_backprop (bool): If this is ``True``, marquetry creates computation graph.
                Otherwise, never create computation graph.
            train (bool): If this is ``True``, marquetry's function computes it as train.
                Otherwise, computes it as predict mode.
            CACHE_DIR (str): The cache directory of the marquetry.
                The default path is ``~/.marquetry``.
    """

    enable_backprop = True
    train = True
    CUDA_ENABLE = cuda_backend.GPU_ENABLE
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".marquetry")

    def show(self, file=sys.stdout):
        keys = sorted(self.__dict__)
        _print_attrs(self, keys, file)


def _print_attrs(obj, keys, file):
    max_len = max(len(key) for key in keys)
    for key in keys:
        spacer = " " * (max_len - len(key))
        print("{}:{}{}".format(key, spacer, getattr(obj, key)), file=file)


config = Config()


@contextlib.contextmanager
def using_config(name, value, config_obj=config):
    """Context manager for temporarily modifying a configuration setting.

        This context manager allows you to temporarily change the value of a configuration setting
        within a specific context. You can use it to override certain configuration parameters for
        a specific block of code while ensuring that the original setting is restored after the block execution.

    Args:
        name (str): The name of the configuration setting to modify.
        value: The new value to set for the specified configuration setting.
        config_obj: The configuration object to modify.
            Defaults to the global configuration object defined in the ``config`` module.

    Yields:
        None

    Examples:
        >>> print(config.train)
        True
        >>> with using_config("train", False):
        >>>     print(config.train)
        False
        >>> print(config.train)
        True

    Note:
        This context manager is typically used in conjunction with
        the global configuration object defined in the 'config' module.
    """

    if hasattr(config_obj, name):
        old_value = getattr(config_obj, name)
        setattr(config_obj, name, value)
        try:
            yield
        finally:
            setattr(config_obj, name, old_value)
    else:
        setattr(config_obj, name, value)
        try:
            yield
        finally:
            delattr(config_obj, name)

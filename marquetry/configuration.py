import os

import contextlib
import sys


class Config(object):
    enable_backprop = True
    train = True
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

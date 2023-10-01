import numpy as np

import marquetry


# ===========================================================================
# Dataset base class
# ===========================================================================
class Dataset(object):
    """Dataset base class that provides data format function.

    All dataset implementations defined in :mod:`marquetry.datasets` inherit this class.


    The main feature of this class is providing data and target following user setting.
    If you specify the :attr:`train` is True, this provides train dataset and if False,
    provides test dataset.

    Args:
        train (bool): toggle if you want to load train_data or not.
        transform (:class:`marquetry.transformers.Compose` or class or None):
            transform means transform **data** that is used when data loaded from anything.
        target_transform(:class:`marquetry.transformers.Compose` or class or None)
            : transform means transform **target** that is used when data loaded from anything.

    Note:
        This object is usually used with DataLoader object.

    """
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.transform is None:
            self.transform = lambda x: x

        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.source = None
        self.target = None

        self._set_data()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.target is None:
            return self.transform(self.source[index]), None
        else:
            return self.transform(self.source[index]), self.target_transform(self.target[index])

    def __len__(self):
        return len(self.source)

    @property
    def source_shape(self):
        return self.source.shape

    def _set_data(self, *args):
        raise NotImplementedError()

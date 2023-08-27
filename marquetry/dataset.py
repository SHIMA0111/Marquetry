import numpy as np


# ===========================================================================
# Dataset base class
# ===========================================================================
class Dataset(object):
    def __init__(self, train=True, transform=None, target_transform=None, **kwargs):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.transform is None:
            self.transform = lambda x: x

        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.source = None
        self.target = None

        self._set_data(**kwargs)

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.target is None:
            return self.transform(self.source[index]), None
        else:
            return self.transform(self.source[index]), self.target_transform(self.target[index])

    def __len__(self):
        return len(self.source)

    def _set_data(self, *args, **kwargs):
        raise NotImplementedError()

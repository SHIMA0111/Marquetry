import gzip

import numpy as np

from marquetry import dataset
from marquetry.utils import get_file
from marquetry.transformers import Flatten


class MNIST(dataset.Dataset):
    """Get the MNIST dataset.

        This dataset is sourced from http://yann.lecun.com/exdb/mnist/.

        The MNIST database of handwritten digits has a training set of 60,000 examples,
        and a test set of 10,000 examples.

        The Attributes and Args is following the :class:`marquetry.dataset.Dataset`, please check it.

        Examples:
            >>> data = MNIST(transform=Flatten())
            >>> data.source.shape
            (60000, 1, 28, 28)
            If the data extruct directly by ``.sorce``, the tranform doesn't apply.
            >>> source, target = data[0]
            >>> source.shape
            (784,)
    """
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(train, transform, target_transform)

    def _set_data(self):
        url = "http://yann.lecun.com/exdb/mnist/"

        train_files = {
            "source": "train-images-idx3-ubyte.gz",
            "target": "train-labels-idx1-ubyte.gz"
        }
        test_files = {
            "source": "t10k-images-idx3-ubyte.gz",
            "target": "t10k-labels-idx1-ubyte.gz"
        }

        files = train_files if self.train else test_files
        source_path = get_file(url + files["source"])
        target_path = get_file(url + files["target"])

        self.source = self._load_source(source_path)
        self.target = self._load_target(target_path)

    @staticmethod
    def _load_source(file_path):
        with gzip.open(file_path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        source = data.reshape((-1, 1, 28, 28))

        return source

    @staticmethod
    def _load_target(file_path):
        with gzip.open(file_path, "rb") as f:
            target = np.frombuffer(f.read(), np.uint8, offset=8)

        return target

    @property
    def labels(self):
        return {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}


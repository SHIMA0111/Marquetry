import gzip

import numpy as np

from marquetry import dataset
from marquetry.utils import get_file
from marquetry.transformers import Compose, Flatten, ToFloat, Normalize


class FashionMNIST(dataset.Dataset):
    """Get the FashionMNIST dataset.

        This dataset is sourced from https://github.com/zalandoresearch/fashion-mnist.

        Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and
        a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

        The Attributes and Args is following the :class:`marquetry.dataset.Dataset`, please check it.

        >>> data = FashionMNIST(transform=Flatten())
        >>> data.source.shape
        (60000, 1, 28, 28)
        If the data extruct directly by ``.sorce``, the tranform doesn't apply.
        >>> source, target = data[0]
        >>> source.shape
        (784,)
        >>>

    """
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(train, transform, target_transform)

    def _set_data(self):
        url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

        train_files = {
            "source": "train-images-idx3-ubyte.gz",
            "target": "train-labels-idx1-ubyte.gz"
        }
        test_files = {
            "source": "t10k-images-idx3-ubyte.gz",
            "target": "t10k-labels-idx1-ubyte.gz"
        }

        files = train_files if self.train else test_files
        class_label = "train" if self.train else "test"
        source_path = get_file(url + files["source"], "fashion_{}_data.gz".format(class_label))
        target_path = get_file(url + files["target"], "fashion_{}_label.gz".format(class_label))

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
        return {
            0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
            4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

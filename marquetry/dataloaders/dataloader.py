import numpy as np

from marquetry import cuda_backend


class DataLoader(object):
    """A utility class for loading and iterating over batches of data.

        The `DataLoader` class is designed to facilitate the loading and processing of
        large datasets in machine learning. It takes a dataset, batch size,
        and optional shuffle flag to create an iterable data loader that provides batches of data.

        Args:
            dataset (:class:`marquetry.dataset.Dataset`): The dataset to be loaded and batched.
            batch_size (int): The size of each batch
            shuffle (bool): Whether to shuffle the data before each epoch.
            cuda (bool): Whether to use GPU (CUDA) for data storage (if available).

        Examples:
            >>> dataset = MyDataset()
            >>> dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            Iterating over batches of data:
            >>> for batch_x, batch_t in dataloader:
            Process the batch of input data (batch_x) and target data (batch_t).

        Caution:
            Generally, DataLoader object is used with ``for`` statement.
            If you have no special reason, you don't need ``reset`` or ``next`` method.
    """
    def __init__(self, dataset, batch_size, shuffle=True, cuda=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cuda = cuda

        self.data_size = len(dataset)
        self.max_iters = self.data_size // batch_size

        self.iterations = 0
        self.index = None

        self.reset()

    def reset(self):
        self.iterations = 0

        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterations >= self.max_iters:
            self.reset()
            raise StopIteration

        batch_index = self.index[self.iterations * self.batch_size:(self.iterations + 1) * self.batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda_backend.cp if self.cuda else np
        x = xp.array([data[0] for data in batch])
        t = xp.array([data[1] for data in batch])

        self.iterations += 1
        return x, t

    def next(self):
        return self.__next__()

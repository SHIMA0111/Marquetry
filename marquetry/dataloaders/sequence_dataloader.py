import numpy as np

from marquetry import cuda_backend
from marquetry import dataloaders


class SeqDataLoader(dataloaders.DataLoader):
    """A specialized data loader for sequential data.

        The `SeqDataLoader` class is a subclass of the `DataLoader` designed specifically for loading sequential data.
        It inherits most of its behavior from the parent `DataLoader` class
        but modifies the batch creation logic to maintain sequence order.

        Args:
            dataset (:class:`marquetry.dataset.Dataset`): The dataset to be loaded and batched.
            batch_size (int): The size of each batch
            cuda (bool): Whether to use GPU (CUDA) for data storage (if available).

        Examples:
            >>> seq_dataset = SequentialDataset()
            >>> seq_dataloader = SeqDataLoader(seq_dataset, batch_size=32, cuda=True)
            Iterating over batches of sequential data while maintaining order:
            >>> for batch_x, batch_t in seq_dataloader:
            Process the batch of input data (batch_x) and target data (batch_t).

        Caution:
            Generally, DataLoader object is used with ``for`` statement.
            If you have no special reason, you don't need ``reset`` or ``next`` method.
    """
    def __init__(self, dataset, batch_size, cuda=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False, cuda=cuda)

    def __next__(self):
        if self.iterations >= self.max_iters:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [(i * jump + self.iterations) % self.data_size for i in range(self.batch_size)]

        batch = [self.dataset[i] for i in batch_index]

        xp = cuda_backend.cp if self.cuda else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iterations += 1

        return x, t

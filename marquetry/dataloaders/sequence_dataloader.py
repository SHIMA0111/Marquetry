import numpy as np

from marquetry import cuda_backend, dataloaders


class SeqDataLoader(dataloaders.DataLoader):
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

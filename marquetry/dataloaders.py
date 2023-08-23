import numpy as np

from marquetry import cuda_backend


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True, cuda=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cuda = cuda

        self.data_size = len(dataset)
        self.max_iters = -(-self.data_size // batch_size)

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


class SeqDataLoader(DataLoader):
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

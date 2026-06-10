import unittest

import numpy as np

import marquetry
from marquetry.dataloaders import DataLoader, SeqDataLoader


class ArangeDataset(marquetry.dataset.Dataset):
    """10 samples: source[i] = [i], target[i] = i."""

    def _set_data(self):
        self.source = np.arange(10, dtype=np.float32).reshape(10, 1)
        self.target = np.arange(10)


class TestDataLoader(unittest.TestCase):

    def test_batch_shapes(self):
        loader = DataLoader(ArangeDataset(), batch_size=3, shuffle=False)

        x, t = next(iter(loader))

        self.assertEqual(x.shape, (3, 1))
        self.assertEqual(t.shape, (3,))

    def test_sequential_order_without_shuffle(self):
        loader = DataLoader(ArangeDataset(), batch_size=3, shuffle=False)

        batches = [t for _, t in loader]
        seen = np.concatenate(batches)

        # 10 = 3 + 3 + 3 + 1: the last partial batch is also returned
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[-1].shape[0], 1)
        self.assertEqual(seen.tolist(), list(range(10)))

    def test_shuffle_covers_all_samples(self):
        loader = DataLoader(ArangeDataset(), batch_size=5, shuffle=True)

        seen = np.concatenate([t for _, t in loader])

        self.assertEqual(sorted(seen.tolist()), list(range(10)))

    def test_second_epoch_works(self):
        loader = DataLoader(ArangeDataset(), batch_size=5, shuffle=False)

        first_epoch = [t for _, t in loader]
        second_epoch = [t for _, t in loader]

        self.assertEqual(len(first_epoch), 2)
        self.assertEqual(len(second_epoch), 2)


class TestSeqDataLoader(unittest.TestCase):

    def test_batch_size_larger_than_dataset_rejected(self):
        with self.assertRaises(ValueError):
            SeqDataLoader(ArangeDataset(), batch_size=11)

    def test_jump_ordering(self):
        loader = SeqDataLoader(ArangeDataset(), batch_size=2)

        x1, t1 = next(loader)
        x2, t2 = next(loader)

        # jump = 10 // 2 = 5: each batch row advances its own stream by one step.
        self.assertEqual(t1.tolist(), [0, 5])
        self.assertEqual(t2.tolist(), [1, 6])
        self.assertEqual(x1.shape, (2, 1))

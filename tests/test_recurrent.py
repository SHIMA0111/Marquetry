import unittest

import numpy as np

import marquetry
import marquetry.functions as funcs
import marquetry.layers as layers
from marquetry import Container
from marquetry.utils import array_close, array_equal


class TestRNN(unittest.TestCase):

    def test_output_shape_and_state(self):
        rnn = layers.RNN(5)
        x = np.random.randn(3, 4).astype(np.float32)

        y1 = rnn(x)

        self.assertEqual(y1.shape, (3, 5))
        self.assertIs(rnn.h, y1)

    def test_state_persists_between_calls(self):
        rnn = layers.RNN(5)
        x = np.random.randn(3, 4).astype(np.float32)

        y1 = rnn(x)
        y2 = rnn(x)

        self.assertFalse(np.allclose(y1.data, y2.data))

    def test_reset_state_reproduces_first_output(self):
        rnn = layers.RNN(5)
        x = np.random.randn(3, 4).astype(np.float32)

        y1 = rnn(x)
        rnn(x)
        rnn.reset_state()
        y3 = rnn(x)

        self.assertTrue(array_equal(y1.data, y3.data))


class TestLSTM(unittest.TestCase):

    def test_output_shape(self):
        lstm = layers.LSTM(6)
        x = np.random.randn(2, 3).astype(np.float32)

        y = lstm(x)

        self.assertEqual(y.shape, (2, 6))

    def test_reset_state_reproduces_first_output(self):
        lstm = layers.LSTM(6)
        x = np.random.randn(2, 3).astype(np.float32)

        y1 = lstm(x)
        lstm(x)
        lstm.reset_state()
        y3 = lstm(x)

        self.assertTrue(array_equal(y1.data, y3.data))

    def test_backward_reaches_all_params(self):
        lstm = layers.LSTM(4)
        x = np.random.randn(2, 3).astype(np.float32)

        # two steps so the hidden-to-gate layers (lazily created) also join the graph
        lstm(x)
        loss = funcs.sum(lstm(x))
        loss.backward()

        grads = [param.grad for param in lstm.params()]
        self.assertTrue(all(grad is not None for grad in grads))


class TestGRU(unittest.TestCase):

    def test_output_shape_and_reset(self):
        gru = layers.GRU(6)
        x = np.random.randn(2, 3).astype(np.float32)

        y1 = gru(x)
        self.assertEqual(y1.shape, (2, 6))

        gru(x)
        gru.reset_state()
        y3 = gru(x)
        self.assertTrue(array_equal(y1.data, y3.data))

    def test_set_state_requires_container(self):
        gru = layers.GRU(6)

        with self.assertRaises(ValueError):
            gru.set_state(np.zeros((2, 6), dtype=np.float32))


class TestBiLSTM(unittest.TestCase):

    def test_output_shape(self):
        bi_lstm = layers.BiLSTM(5)
        x = np.random.randn(3, 7, 4).astype(np.float32)

        y = bi_lstm(x)

        self.assertEqual(y.shape, (3, 7, 10))

    def test_each_call_processes_sequence_independently(self):
        bi_lstm = layers.BiLSTM(5)
        x = np.random.randn(3, 6, 4).astype(np.float32)

        y1 = bi_lstm(x)
        y2 = bi_lstm(x)

        self.assertTrue(array_equal(y1.data, y2.data))

    def test_rejects_non_sequence_input(self):
        bi_lstm = layers.BiLSTM(5)

        with self.assertRaises(ValueError):
            bi_lstm(np.random.randn(3, 4).astype(np.float32))

    def test_backward_reaches_all_params(self):
        bi_lstm = layers.BiLSTM(4)
        x = np.random.randn(2, 5, 3).astype(np.float32)

        loss = funcs.sum(bi_lstm(x))
        loss.backward()

        grads = [param.grad for param in bi_lstm.params()]
        self.assertTrue(all(grad is not None for grad in grads))


class TestEmbedding(unittest.TestCase):

    def test_lookup(self):
        embedding = layers.Embedding(10, 4)
        indexes = np.array([0, 3, 3, 7])

        y = embedding(indexes)

        self.assertTrue(array_equal(y.data, embedding.w.data[indexes]))

    def test_gradient_scatters_to_rows(self):
        embedding = layers.Embedding(10, 4)
        indexes = np.array([0, 3, 3])

        loss = funcs.sum(embedding(indexes))
        loss.backward()

        grad = embedding.w.grad.data
        self.assertTrue(array_close(grad[0], np.ones(4)))
        self.assertTrue(array_close(grad[3], np.full(4, 2.0)))
        self.assertTrue(array_close(grad[1], np.zeros(4)))

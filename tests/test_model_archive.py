import json
import os
import tempfile
import unittest
import zipfile

import numpy as np

import marquetry
import marquetry.functions as funcs
import marquetry.layers as layers
from marquetry import Model, optimizers
from marquetry.models import CNN, MLP, Sequential
from marquetry.model_archive import (
    ArchiveError, FORMAT_VERSION, GraphModel, load_archive, save_archive)


class FuncModel(Model):

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, *inputs):
        return self._fn(*inputs)


def random_array(*shape):
    return np.random.randn(*shape).astype(np.float32)


def roundtrip(model, inputs, **save_kwargs):
    """Save the model to a temporary archive and load it back."""
    if not isinstance(inputs, (tuple, list)):
        inputs = [inputs]

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "model.mq")
        save_archive(model, inputs, path, **save_kwargs)
        restored = load_archive(path)

    return restored


def assert_roundtrip(testcase, model, inputs):
    """The restored graph replays the same functions, so outputs match exactly."""
    if not isinstance(inputs, (tuple, list)):
        inputs = [inputs]

    with marquetry.test_mode():
        expected = model(*[marquetry.as_container(np.copy(x)) for x in inputs])
    expected = expected if isinstance(expected, tuple) else (expected,)

    restored = roundtrip(model, [np.copy(x) for x in inputs])

    with marquetry.test_mode():
        actual = restored(*inputs)
    actual = actual if isinstance(actual, tuple) else (actual,)

    testcase.assertEqual(len(actual), len(expected))
    for got, exp in zip(actual, expected):
        np.testing.assert_array_equal(got.data, exp.data)

    return restored


class TestOperatorRoundTrip(unittest.TestCase):

    def test_linear(self):
        assert_roundtrip(self, Sequential(layers.Linear(7)), random_array(4, 3))

    def test_linear_without_bias(self):
        assert_roundtrip(self, Sequential(layers.Linear(7, nobias=True)), random_array(4, 3))

    def test_convolution(self):
        assert_roundtrip(
            self, Sequential(layers.Convolution2D(4, (3, 3), stride=2, pad=1)),
            random_array(2, 3, 9, 9))

    def test_deconvolution(self):
        assert_roundtrip(
            self, Sequential(layers.Deconvolution2D(4, (3, 3), stride=2, pad=1)),
            random_array(2, 3, 5, 5))

    def test_max_pooling_with_pad(self):
        assert_roundtrip(
            self, FuncModel(lambda x: funcs.max_pooling_2d(x, (3, 3), stride=2, pad=1)),
            random_array(2, 3, 7, 7) - 5.0)

    def test_batch_normalization_with_trained_stats(self):
        model = Sequential(layers.BatchNormalization())
        for _ in range(3):
            model(random_array(8, 5) * 2.0 + 1.0)
        assert_roundtrip(self, model, random_array(4, 5))

    def test_layer_normalization(self):
        assert_roundtrip(self, Sequential(layers.LayerNormalization()), random_array(4, 6))

    def test_l2_normalization(self):
        assert_roundtrip(
            self, FuncModel(lambda x: funcs.l2_normalization(x, axis=1)), random_array(4, 6))

    def test_activations(self):
        assert_roundtrip(
            self,
            FuncModel(lambda x: funcs.swish(funcs.gelu(funcs.relu(x), approximate="sigmoid"),
                                            beta=2.0)),
            random_array(4, 6))

    def test_prelu_layer(self):
        assert_roundtrip(
            self, Sequential(layers.PReLU(num_parameter=3, init=0.3)),
            random_array(2, 3, 5, 5))

    def test_glu(self):
        assert_roundtrip(self, FuncModel(funcs.glu), random_array(4, 6))

    def test_scalar_constant_is_captured(self):
        assert_roundtrip(self, FuncModel(lambda x: (x + 2.0) * 0.5), random_array(4, 5))

    def test_math_chain(self):
        assert_roundtrip(
            self,
            FuncModel(lambda x: funcs.clip(x ** 3, -0.5, 0.5) / (funcs.exp(-x) + 2.0)),
            random_array(4, 5))

    def test_reductions(self):
        assert_roundtrip(
            self,
            FuncModel(lambda x: funcs.sum(x, axis=1, keepdims=True)
                      + funcs.mean(x, axis=1, keepdims=True)),
            random_array(3, 4))

    def test_array_operations(self):
        assert_roundtrip(
            self,
            FuncModel(lambda x: funcs.concat(
                (funcs.transpose(x, (1, 0, 2)), funcs.transpose(x, (1, 0, 2))), axis=2)),
            random_array(2, 3, 4))

    def test_repeat_unsupported_by_onnx_works_here(self):
        assert_roundtrip(self, FuncModel(lambda x: funcs.repeat(x, 2, 0)), random_array(3, 4))

    def test_get_item(self):
        assert_roundtrip(self, FuncModel(lambda x: x[:, 1:3]), random_array(4, 5))
        assert_roundtrip(self, FuncModel(lambda x: x[1]), random_array(4, 5))
        assert_roundtrip(self, FuncModel(lambda x: x[::2]), random_array(5, 4))

    def test_split_multi_output(self):
        assert_roundtrip(
            self, FuncModel(lambda x: tuple(funcs.split(x, 2, axis=1))), random_array(4, 6))

    def test_multi_input(self):
        assert_roundtrip(
            self, FuncModel(lambda a, b: funcs.matmul(a, b) + 1.0),
            [random_array(4, 5), random_array(5, 3)])

    def test_dropout_traces_and_replays(self):
        assert_roundtrip(self, FuncModel(lambda x: funcs.dropout(x, 0.5)), random_array(4, 5))


class TestModelRoundTrip(unittest.TestCase):

    def test_mlp(self):
        model = MLP([16, 8], activation=funcs.relu, is_dropout=True)
        assert_roundtrip(self, model, random_array(4, 12))

    def test_cnn(self):
        assert_roundtrip(self, CNN(10), random_array(2, 1, 12, 12))

    def test_sequential_mixed(self):
        model = Sequential(
            layers.Convolution2D(4, (3, 3), pad=1),
            layers.BatchNormalization(),
            funcs.relu,
            funcs.flatten,
            layers.Linear(5),
        )
        model(random_array(4, 3, 6, 6))  # one train-mode pass to give BN real statistics
        assert_roundtrip(self, model, random_array(2, 3, 6, 6))

    def test_dynamic_batch(self):
        model = CNN(7)
        restored = assert_roundtrip(self, model, random_array(2, 1, 12, 12))

        for batch_size in (1, 5):
            other = random_array(batch_size, 1, 12, 12)
            with marquetry.test_mode():
                expected = model(other)
                actual = restored(other)
            np.testing.assert_array_equal(actual.data, expected.data)

    def test_static_batch_keeps_literal_reshape(self):
        model = CNN(7)
        restored = roundtrip(model, random_array(2, 1, 12, 12), dynamic_batch=False)

        with marquetry.test_mode():
            restored(random_array(2, 1, 12, 12))  # traced batch size still works
            with self.assertRaises(ValueError):
                restored(random_array(3, 1, 12, 12))


class TestRestoredModelBehavior(unittest.TestCase):

    def test_parameters_are_trainable(self):
        # Deterministic data: with a fixed seed, every parameter tensor receives a
        # non-zero gradient, so asserting that all of them change is stable.
        np.random.seed(0)
        restored = roundtrip(MLP([8, 4], activation=funcs.relu, is_dropout=False),
                             random_array(4, 6))

        params_before = {id(p): np.copy(p.data) for p in restored.params()}
        self.assertTrue(params_before)

        optimizer = optimizers.Adam().prepare(restored)
        x = random_array(16, 6)
        t = np.random.randint(0, 4, size=16)

        y = restored(x)
        loss = funcs.softmax_cross_entropy(y, t)
        restored.clear_grads()
        loss.backward()
        optimizer.update()

        changed = [not np.array_equal(params_before[id(p)], p.data)
                   for p in restored.params()]
        self.assertTrue(all(changed))

    def test_train_test_mode_dispatch_is_preserved(self):
        restored = roundtrip(FuncModel(lambda x: funcs.dropout(x, 0.5)), random_array(4, 6))
        x = np.ones((64, 16), dtype=np.float32)

        with marquetry.test_mode():
            test_out = restored(x).data
        np.testing.assert_array_equal(test_out, x)

        train_out = restored(x).data  # train mode: dropout randomly zeroes entries
        self.assertTrue((train_out == 0.0).any())

    def test_save_params_keys_match_original(self):
        model = MLP([8, 4], activation=funcs.relu)
        model(random_array(2, 6))  # initialize the lazy weights
        restored = roundtrip(model, random_array(2, 6))

        original_params, restored_params = {}, {}
        model._flatten_params(original_params)
        restored._flatten_params(restored_params)

        original_keys = {key for key, param in original_params.items() if param is not None}
        self.assertEqual(original_keys, set(restored_params.keys()))


class TestArchiveFile(unittest.TestCase):

    def test_archive_is_a_zip_with_expected_members(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.mq")
            save_archive(Sequential(layers.Linear(3)), random_array(4, 5), path)

            with zipfile.ZipFile(path) as archive:
                members = set(archive.namelist())
                self.assertEqual(members, {"format.json", "graph.json", "weights.npz"})

                header = json.loads(archive.read("format.json"))
                self.assertEqual(header["format"], "marquetry_archive")
                self.assertEqual(header["version"], FORMAT_VERSION)

                graph = json.loads(archive.read("graph.json"))
                self.assertEqual([node["op"] for node in graph["nodes"]], ["Linear"])
                self.assertEqual(len(graph["parameters"]), 2)

    def test_load_returns_graph_model(self):
        restored = roundtrip(Sequential(layers.Linear(3)), random_array(4, 5))
        self.assertIsInstance(restored, GraphModel)
        self.assertIsInstance(restored, Model)

    def test_unsupported_function_raises_with_name(self):
        model = FuncModel(lambda x: funcs.softmax_cross_entropy(x, np.array([0, 1])))
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.mq")
            with self.assertRaises(ArchiveError) as raised:
                save_archive(model, random_array(2, 3), path)

        self.assertIn("SoftmaxCrossEntropy", str(raised.exception))

    def test_future_version_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.mq")
            save_archive(Sequential(layers.Linear(3)), random_array(4, 5), path)

            tampered = os.path.join(tmp_dir, "tampered.mq")
            with zipfile.ZipFile(path) as source, zipfile.ZipFile(tampered, "w") as target:
                for member in source.namelist():
                    payload = source.read(member)
                    if member == "format.json":
                        header = json.loads(payload)
                        header["version"] = FORMAT_VERSION + 999
                        payload = json.dumps(header)
                    target.writestr(member, payload)

            with self.assertRaises(ArchiveError):
                load_archive(tampered)

    def test_batch_dim_detected_and_resolved_inside_slice(self):
        from marquetry.model_archive import spec

        value = (slice(spec.BATCH_DIM, None, None), 3)
        self.assertTrue(spec.contains_batch_dim(value))
        self.assertEqual(spec.resolve_batch_dim(value, 8), (slice(8, None, None), 3))

    def test_not_an_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "not_archive.mq")
            with open(path, "wb") as f:
                f.write(b"not a zip at all")

            with self.assertRaises(ArchiveError):
                load_archive(path)


if __name__ == "__main__":
    unittest.main()

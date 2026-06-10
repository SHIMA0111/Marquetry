import unittest

import numpy as np
import onnxruntime

import marquetry
import marquetry.functions as funcs
import marquetry.layers as layers
from marquetry import Model
from marquetry.models import CNN, MLP, Sequential
from marquetry.onnx_export import export_onnx


class FuncModel(Model):
    """Wrap a plain callable so function-level graphs can be exported and compared."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, *inputs):
        return self._fn(*inputs)


def run_onnx(proto, inputs):
    session = onnxruntime.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"])
    input_metas = session.get_inputs()
    if len(input_metas) != len(inputs):
        raise AssertionError(
            "the exported model expects {} inputs, but {} were provided."
            .format(len(input_metas), len(inputs)))
    feeds = {meta.name: np.asarray(data) for meta, data in zip(input_metas, inputs)}

    return session.run(None, feeds)


def assert_equivalent(testcase, model, inputs, rtol=1e-5, atol=1e-6, **export_kwargs):
    """Export the model and compare onnxruntime outputs with marquetry's test-mode forward."""
    if not isinstance(inputs, (tuple, list)):
        inputs = [inputs]

    with marquetry.test_mode():
        expected = model(*[marquetry.as_container(np.copy(x)) for x in inputs])
    expected = expected if isinstance(expected, tuple) else (expected,)

    proto = export_onnx(model, [np.copy(x) for x in inputs], **export_kwargs)
    actual = run_onnx(proto, inputs)

    testcase.assertEqual(len(actual), len(expected))
    for got, exp in zip(actual, expected):
        np.testing.assert_allclose(got, exp.data, rtol=rtol, atol=atol)

    return proto


def random_array(*shape):
    return np.random.randn(*shape).astype(np.float32)


class TestLinear(unittest.TestCase):

    def test_linear_with_bias(self):
        assert_equivalent(self, Sequential(layers.Linear(7)), random_array(4, 3))

    def test_linear_without_bias(self):
        assert_equivalent(self, Sequential(layers.Linear(7, nobias=True)), random_array(4, 3))

    def test_linear_3d_input(self):
        assert_equivalent(self, Sequential(layers.Linear(6)), random_array(2, 5, 3))

    def test_linear_3d_input_without_bias(self):
        assert_equivalent(
            self, Sequential(layers.Linear(6, nobias=True)), random_array(2, 5, 3))


class TestConvolutionAndPooling(unittest.TestCase):

    def test_convolution(self):
        assert_equivalent(
            self, Sequential(layers.Convolution2D(8, (3, 3))), random_array(2, 3, 8, 8),
            rtol=1e-4, atol=1e-5)

    def test_convolution_stride_pad(self):
        assert_equivalent(
            self, Sequential(layers.Convolution2D(4, (3, 3), stride=2, pad=1)),
            random_array(2, 3, 9, 9), rtol=1e-4, atol=1e-5)

    def test_convolution_without_bias(self):
        assert_equivalent(
            self, Sequential(layers.Convolution2D(4, (3, 3), nobias=True)),
            random_array(2, 3, 8, 8), rtol=1e-4, atol=1e-5)

    def test_deconvolution(self):
        assert_equivalent(
            self, Sequential(layers.Deconvolution2D(4, (3, 3), stride=2, pad=1)),
            random_array(2, 3, 5, 5), rtol=1e-4, atol=1e-5)

    def test_max_pooling(self):
        assert_equivalent(
            self, FuncModel(lambda x: funcs.max_pooling_2d(x, (2, 2))),
            random_array(2, 3, 8, 8))

    def test_max_pooling_with_pad(self):
        # Negative inputs make the zero-padding visible if it is mistranslated.
        x = random_array(2, 3, 7, 7) - 5.0
        assert_equivalent(
            self, FuncModel(lambda x: funcs.max_pooling_2d(x, (3, 3), stride=2, pad=1)), x)


class TestNormalization(unittest.TestCase):

    def _trained_batch_norm(self, sample_shape):
        model = Sequential(layers.BatchNormalization())
        for _ in range(3):
            model(random_array(*sample_shape) * 2.0 + 1.0)

        return model

    def test_batch_normalization_2d(self):
        model = self._trained_batch_norm((8, 5))
        assert_equivalent(self, model, random_array(4, 5))

    def test_batch_normalization_4d(self):
        model = self._trained_batch_norm((4, 3, 6, 6))
        assert_equivalent(self, model, random_array(2, 3, 6, 6))

    def test_layer_normalization_2d(self):
        assert_equivalent(
            self, Sequential(layers.LayerNormalization()), random_array(4, 6),
            rtol=1e-4, atol=1e-5)

    def test_layer_normalization_4d(self):
        assert_equivalent(
            self, Sequential(layers.LayerNormalization()), random_array(2, 3, 4, 4),
            rtol=1e-4, atol=1e-5)

    def test_l2_normalization(self):
        assert_equivalent(
            self, FuncModel(lambda x: funcs.l2_normalization(x, axis=1)), random_array(4, 6))


class TestActivations(unittest.TestCase):

    def _check(self, fn, x=None, **kwargs):
        if x is None:
            x = random_array(4, 6)
        assert_equivalent(self, FuncModel(fn), x, **kwargs)

    def test_relu(self):
        self._check(funcs.relu)

    def test_leaky_relu(self):
        self._check(lambda x: funcs.leaky_relu(x, slope=0.1))

    def test_sigmoid(self):
        self._check(funcs.sigmoid)

    def test_tanh(self):
        self._check(funcs.tanh)

    def test_softmax_axis1(self):
        self._check(lambda x: funcs.softmax(x, axis=1))

    def test_softmax_last_axis(self):
        self._check(lambda x: funcs.softmax(x, axis=2), x=random_array(2, 3, 5))

    def test_log_softmax(self):
        self._check(lambda x: funcs.log_softmax(x, axis=1))

    def test_softplus(self):
        self._check(funcs.softplus)

    def test_softplus_beta(self):
        self._check(lambda x: funcs.softplus(x, beta=2))

    def test_mish(self):
        self._check(funcs.mish)

    def test_gelu_exact(self):
        self._check(lambda x: funcs.gelu(x, approximate="none"))

    def test_gelu_tanh(self):
        self._check(lambda x: funcs.gelu(x, approximate="tanh"))

    def test_gelu_sigmoid(self):
        self._check(lambda x: funcs.gelu(x, approximate="sigmoid"))

    def test_gelu_exact_decomposed_for_opset18(self):
        self._check(lambda x: funcs.gelu(x, approximate="none"), opset_version=18)

    def test_gelu_tanh_decomposed_for_opset18(self):
        self._check(lambda x: funcs.gelu(x, approximate="tanh"), opset_version=18)

    def test_glu(self):
        self._check(funcs.glu)

    def test_swish(self):
        self._check(funcs.swish)

    def test_swish_beta(self):
        self._check(lambda x: funcs.swish(x, beta=2.0))

    def test_identity(self):
        self._check(funcs.identity)

    def test_prelu_layer_2d(self):
        model = Sequential(layers.PReLU(num_parameter=6, init=0.3))
        assert_equivalent(self, model, random_array(4, 6))

    def test_prelu_layer_4d(self):
        model = Sequential(layers.PReLU(num_parameter=3, init=0.3))
        assert_equivalent(self, model, random_array(2, 3, 5, 5))

    def test_dynamic_swish_layer(self):
        assert_equivalent(self, Sequential(layers.DynamicSwish()), random_array(4, 6))

    def test_dropout_is_inference_identity(self):
        self._check(lambda x: funcs.dropout(x, 0.5))


class TestMath(unittest.TestCase):

    def _check(self, fn, x=None, **kwargs):
        if x is None:
            x = random_array(4, 5)
        assert_equivalent(self, FuncModel(fn), x, **kwargs)

    def test_add_tensors(self):
        assert_equivalent(
            self, FuncModel(lambda a, b: a + b), [random_array(4, 5), random_array(4, 5)])

    def test_add_broadcast(self):
        assert_equivalent(
            self, FuncModel(lambda a, b: a + b), [random_array(4, 5), random_array(5)])

    def test_add_scalar(self):
        self._check(lambda x: x + 2.0)

    def test_sub(self):
        assert_equivalent(
            self, FuncModel(lambda a, b: a - b), [random_array(4, 5), random_array(4, 5)])

    def test_mul(self):
        assert_equivalent(
            self, FuncModel(lambda a, b: a * b), [random_array(4, 5), random_array(4, 5)])

    def test_div(self):
        denominator = np.abs(random_array(4, 5)) + 1.0
        assert_equivalent(
            self, FuncModel(lambda a, b: a / b), [random_array(4, 5), denominator])

    def test_neg(self):
        self._check(lambda x: -x)

    def test_pow(self):
        self._check(lambda x: x ** 3)

    def test_absolute(self):
        self._check(funcs.absolute)

    def test_exp(self):
        self._check(funcs.exp)

    def test_log(self):
        self._check(funcs.log, x=np.abs(random_array(4, 5)) + 0.5)

    def test_log2(self):
        self._check(funcs.log2, x=np.abs(random_array(4, 5)) + 0.5)

    def test_log10(self):
        self._check(funcs.log10, x=np.abs(random_array(4, 5)) + 0.5)

    def test_sqrt(self):
        self._check(funcs.sqrt, x=np.abs(random_array(4, 5)) + 0.5)

    def test_square(self):
        self._check(funcs.square)

    def test_clip(self):
        self._check(lambda x: funcs.clip(x, -0.5, 0.5))

    def test_matmul(self):
        assert_equivalent(
            self, FuncModel(funcs.matmul), [random_array(4, 5), random_array(5, 3)])

    def test_chained_expression(self):
        self._check(lambda x: (x * 2.0 + 1.0) / (funcs.exp(-x) + 2.0))


class TestReductions(unittest.TestCase):

    def _check(self, fn, x=None, **kwargs):
        if x is None:
            x = random_array(3, 4, 5)
        assert_equivalent(self, FuncModel(fn), x, **kwargs)

    def test_sum_all(self):
        self._check(lambda x: funcs.sum(x))

    def test_sum_axis_keepdims(self):
        self._check(lambda x: funcs.sum(x, axis=1, keepdims=True))

    def test_mean(self):
        self._check(lambda x: funcs.mean(x, axis=2))

    def test_max(self):
        self._check(lambda x: funcs.max(x, axis=1))

    def test_min(self):
        self._check(lambda x: funcs.min(x, axis=(0, 2), keepdims=True))


class TestArrayOperations(unittest.TestCase):

    def _check(self, fn, x=None, **kwargs):
        if x is None:
            x = random_array(2, 3, 4)
        assert_equivalent(self, FuncModel(fn), x, **kwargs)

    def test_reshape(self):
        self._check(lambda x: funcs.reshape(x, (2, 12)))

    def test_flatten(self):
        self._check(funcs.flatten)

    def test_transpose_default(self):
        self._check(lambda x: funcs.transpose(x))

    def test_transpose_perm(self):
        self._check(lambda x: funcs.transpose(x, (1, 0, 2)))

    def test_concat(self):
        assert_equivalent(
            self, FuncModel(lambda a, b: funcs.concat((a, b), axis=1)),
            [random_array(2, 3), random_array(2, 4)])

    def test_squeeze(self):
        self._check(lambda x: funcs.squeeze(x, axis=1), x=random_array(3, 1, 4))

    def test_unsqueeze(self):
        self._check(lambda x: funcs.unsqueeze(x, 1), x=random_array(3, 4))

    def test_broadcast_to(self):
        self._check(lambda x: funcs.broadcast_to(x, (2, 3, 4)), x=random_array(3, 4))

    def test_split_sections(self):
        x = random_array(4, 6)
        model = FuncModel(lambda x: tuple(funcs.split(x, 2, axis=1)))
        proto = assert_equivalent(self, model, x)
        self.assertEqual(len(proto.graph.output), 2)

    def test_get_item_slices(self):
        self._check(lambda x: x[:, 1:3])

    def test_get_item_integer(self):
        self._check(lambda x: x[1])

    def test_get_item_step(self):
        self._check(lambda x: x[::2])


class TestModels(unittest.TestCase):

    def test_mlp_with_dropout(self):
        model = MLP([16, 8], activation=funcs.relu, is_dropout=True)
        assert_equivalent(self, model, random_array(4, 12))

    def test_mlp_sigmoid_without_dropout(self):
        model = MLP([16, 8], activation=funcs.sigmoid, is_dropout=False)
        assert_equivalent(self, model, random_array(4, 12))

    def test_cnn(self):
        model = CNN(10)
        assert_equivalent(self, model, random_array(2, 1, 12, 12), rtol=1e-4, atol=1e-5)

    def test_sequential_mixed(self):
        model = Sequential(
            layers.Convolution2D(4, (3, 3), pad=1),
            layers.BatchNormalization(),
            funcs.relu,
            funcs.flatten,
            layers.Linear(5),
        )
        model(random_array(4, 3, 6, 6))  # one train-mode pass to give BN real statistics
        assert_equivalent(self, model, random_array(2, 3, 6, 6), rtol=1e-4, atol=1e-5)

    def test_dynamic_batch_runs_other_batch_sizes(self):
        model = CNN(7)
        traced = random_array(2, 1, 12, 12)
        proto = assert_equivalent(self, model, traced, rtol=1e-4, atol=1e-5)

        session = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CPUExecutionProvider"])
        for batch_size in (1, 5):
            other = random_array(batch_size, 1, 12, 12)
            with marquetry.test_mode():
                expected = model(other)
            got = session.run(None, {"input": other})[0]
            np.testing.assert_allclose(got, expected.data, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

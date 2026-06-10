import os
import tempfile
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

import marquetry
import marquetry.functions as funcs
import marquetry.layers as layers
from marquetry import Model
from marquetry.models import MLP, Sequential
from marquetry.onnx_export import export_onnx, ONNXExportError
from marquetry.onnx_export.exporter import TARGET_IR_VERSION


class FuncModel(Model):

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, *inputs):
        return self._fn(*inputs)


def random_array(*shape):
    return np.random.randn(*shape).astype(np.float32)


class TestVersionPinning(unittest.TestCase):

    def test_default_opset_and_ir_version(self):
        proto = export_onnx(Sequential(layers.Linear(3)), random_array(4, 5))

        default_domain = [opset for opset in proto.opset_import if opset.domain == ""]
        self.assertEqual(len(default_domain), 1)
        self.assertEqual(default_domain[0].version, 21)
        self.assertEqual(proto.ir_version, TARGET_IR_VERSION)

    def test_custom_opset_version(self):
        proto = export_onnx(Sequential(layers.Linear(3)), random_array(4, 5), opset_version=18)
        self.assertEqual(proto.opset_import[0].version, 18)

    def test_too_old_opset_rejected(self):
        with self.assertRaises(ValueError):
            export_onnx(Sequential(layers.Linear(3)), random_array(4, 5), opset_version=17)

    def test_producer_name(self):
        proto = export_onnx(Sequential(layers.Linear(3)), random_array(4, 5))
        self.assertEqual(proto.producer_name, "marquetry")


class TestGraphStructure(unittest.TestCase):

    def test_weights_are_initializers_not_inputs(self):
        proto = export_onnx(Sequential(layers.Linear(3)), random_array(4, 5))

        input_names = [value_info.name for value_info in proto.graph.input]
        self.assertEqual(input_names, ["input"])

        initializer_names = {tensor.name for tensor in proto.graph.initializer}
        self.assertEqual(len(initializer_names), 2)  # weight and bias
        self.assertFalse(initializer_names & set(input_names))

    def test_no_dropout_node_in_inference_graph(self):
        model = MLP([8, 4], activation=funcs.relu, is_dropout=True)
        proto = export_onnx(model, random_array(4, 6))

        op_types = [node.op_type for node in proto.graph.node]
        self.assertNotIn("Dropout", op_types)

    def test_gemm_used_for_2d_linear(self):
        proto = export_onnx(Sequential(layers.Linear(3)), random_array(4, 5))
        self.assertIn("Gemm", [node.op_type for node in proto.graph.node])

    def test_matmul_used_for_3d_linear(self):
        proto = export_onnx(Sequential(layers.Linear(3)), random_array(2, 4, 5))

        op_types = [node.op_type for node in proto.graph.node]
        self.assertNotIn("Gemm", op_types)
        self.assertIn("MatMul", op_types)
        self.assertIn("Add", op_types)

    def test_output_names_are_unique_ssa(self):
        model = MLP([8, 8, 4], activation=funcs.relu)
        proto = export_onnx(model, random_array(4, 6))

        produced = [name for node in proto.graph.node for name in node.output]
        self.assertEqual(len(produced), len(set(produced)))

    def test_multi_output_graph(self):
        model = FuncModel(lambda x: tuple(funcs.split(x, 2, axis=1)))
        proto = export_onnx(model, random_array(4, 6))

        self.assertEqual(len(proto.graph.output), 2)
        self.assertEqual([out.name for out in proto.graph.output], ["output_0", "output_1"])

    def test_multi_input_graph(self):
        model = FuncModel(lambda a, b: a + b)
        proto = export_onnx(model, [random_array(4, 5), random_array(4, 5)])

        self.assertEqual([value_info.name for value_info in proto.graph.input],
                         ["input_0", "input_1"])

    def test_parameter_names_follow_layer_hierarchy(self):
        proto = export_onnx(Sequential(layers.Linear(3)), random_array(4, 5))

        initializer_names = {tensor.name for tensor in proto.graph.initializer}
        self.assertIn("l0/w", initializer_names)
        self.assertIn("l0/b", initializer_names)


class TestDynamicBatch(unittest.TestCase):

    def test_batch_axis_is_symbolic(self):
        proto = export_onnx(Sequential(layers.Linear(3)), random_array(4, 5))

        input_dim = proto.graph.input[0].type.tensor_type.shape.dim
        self.assertEqual(input_dim[0].dim_param, "batch_size")
        self.assertEqual(input_dim[1].dim_value, 5)

        output_dim = proto.graph.output[0].type.tensor_type.shape.dim
        self.assertEqual(output_dim[0].dim_param, "batch_size")

    def test_static_batch_when_disabled(self):
        proto = export_onnx(
            Sequential(layers.Linear(3)), random_array(4, 5), dynamic_batch=False)

        input_dim = proto.graph.input[0].type.tensor_type.shape.dim
        self.assertEqual(input_dim[0].dim_value, 4)

    def test_flatten_reshape_keeps_batch_dynamic(self):
        model = FuncModel(funcs.flatten)
        proto = export_onnx(model, random_array(2, 3, 4))

        shape_tensors = [tensor for tensor in proto.graph.initializer
                         if tensor.name.startswith("reshape_shape")]
        self.assertEqual(list(numpy_helper.to_array(shape_tensors[0])), [0, -1])

    def test_coincidental_batch_sized_reshape_stays_literal(self):
        # After the transpose, the leading axis is no longer the batch axis,
        # so the literal 2 must not be rewritten even though it equals the
        # traced batch size.
        model = FuncModel(lambda x: funcs.reshape(funcs.transpose(x), (2, 12)))
        proto = export_onnx(model, random_array(2, 12))

        shape_tensors = [tensor for tensor in proto.graph.initializer
                         if tensor.name.startswith("reshape_shape")]
        self.assertEqual(list(numpy_helper.to_array(shape_tensors[0])), [2, 12])


class TestExportInterface(unittest.TestCase):

    def test_save_to_file_and_reload(self):
        model = Sequential(layers.Linear(3))
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.onnx")
            export_onnx(model, random_array(4, 5), path)

            self.assertTrue(os.path.exists(path))
            loaded = onnx.load(path)
            onnx.checker.check_model(loaded)

    def test_model_method(self):
        model = MLP([8, 4], activation=funcs.relu)
        proto = model.export_onnx(random_array(4, 6))
        onnx.checker.check_model(proto)

    def test_custom_io_names(self):
        proto = export_onnx(
            Sequential(layers.Linear(3)), random_array(4, 5),
            input_names=["features"], output_names=["logits"])

        self.assertEqual(proto.graph.input[0].name, "features")
        self.assertEqual(proto.graph.output[0].name, "logits")

    def test_wrong_io_name_count_rejected(self):
        with self.assertRaises(ValueError):
            export_onnx(Sequential(layers.Linear(3)), random_array(4, 5),
                        input_names=["a", "b"])

    def test_container_sample_input(self):
        sample = marquetry.array(random_array(4, 5))
        proto = export_onnx(Sequential(layers.Linear(3)), sample)
        onnx.checker.check_model(proto)


class TestUnsupportedCases(unittest.TestCase):

    def test_unsupported_function_raises_with_name(self):
        model = FuncModel(lambda x: funcs.repeat(x, 2, 0))
        with self.assertRaises(ONNXExportError) as raised:
            export_onnx(model, random_array(4, 5))

        self.assertIn("Repeat", str(raised.exception))

    def test_graphless_model_rejected(self):
        model = FuncModel(lambda x: marquetry.array(np.zeros((2, 2), dtype=np.float32)))
        with self.assertRaises(ONNXExportError):
            export_onnx(model, random_array(4, 5))

    def test_empty_inputs_rejected(self):
        with self.assertRaises(ValueError):
            export_onnx(Sequential(layers.Linear(3)), [])


if __name__ == "__main__":
    unittest.main()

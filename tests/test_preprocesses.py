import tempfile
import unittest

import numpy as np
import pandas as pd

import marquetry
from marquetry.preprocesses import ColumnNormalize, LabelEncode, MissImputation, OneHotEncode


class PreprocessTestCase(unittest.TestCase):
    """Isolate each test's statistic cache in a temporary directory."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._config_context = marquetry.using_config("CACHE_DIR", self._tmp_dir.name)
        self._config_context.__enter__()

    def tearDown(self):
        self._config_context.__exit__(None, None, None)
        self._tmp_dir.cleanup()


class TestLabelEncode(PreprocessTestCase):

    def test_encodes_to_integers(self):
        data = pd.DataFrame({"c1": ["a", "b", "a", "c"]})
        encoder = LabelEncode(["c1"], "test_label_basic", True)

        result = encoder(data)

        self.assertEqual(len(set(result["c1"])), 3)
        self.assertTrue(all(isinstance(v, (int, np.integer)) for v in result["c1"]))

    def test_inference_reuses_train_mapping(self):
        train = pd.DataFrame({"c1": ["a", "b", "a"]})
        train_encoder = LabelEncode(["c1"], "test_label_reuse", True)
        train_result = train_encoder(train)

        code_a = train_result["c1"].iloc[0]
        code_b = train_result["c1"].iloc[1]

        test = pd.DataFrame({"c1": ["b", "a", "b"]})
        test_encoder = LabelEncode(["c1"], "test_label_reuse", False)
        test_result = test_encoder(test)

        self.assertEqual(test_result["c1"].tolist(), [code_b, code_a, code_b])

    def test_raise_error_mode_on_unknown(self):
        train = pd.DataFrame({"c1": ["a", "b", "a"]})
        LabelEncode(["c1"], "test_label_raise", True)(train)

        test = pd.DataFrame({"c1": ["a", "z", "b"]})
        encoder = LabelEncode(["c1"], "test_label_raise", False, treat_unknown="raise_error")

        with self.assertRaises(ValueError):
            encoder(test)


class TestOneHotEncode(PreprocessTestCase):

    def test_drop_first_encoding(self):
        data = pd.DataFrame({"c1": ["a", "b", "c", "a"]})
        encoder = OneHotEncode(["c1"], "test_onehot_basic", True)

        result = encoder(data)

        # 3 categories with drop-first -> 2 columns, each row has at most one 1
        self.assertEqual(result.shape, (4, 2))
        self.assertTrue((result.sum(axis=1) <= 1).all())
        self.assertTrue(set(np.unique(result.to_numpy())).issubset({0, 1}))

    def test_inference_columns_consistent(self):
        train = pd.DataFrame({"c1": ["a", "b", "c", "a"]})
        train_encoder = OneHotEncode(["c1"], "test_onehot_consistency", True)
        train_result = train_encoder(train)

        test = pd.DataFrame({"c1": ["c", "a"]})
        test_encoder = OneHotEncode(["c1"], "test_onehot_consistency", False)
        test_result = test_encoder(test)

        self.assertEqual(list(test_result.columns), list(train_result.columns))


class TestMissImputation(PreprocessTestCase):

    def test_category_mode_imputation(self):
        data = pd.DataFrame({"cat": ["x", "x", np.nan, "y"]})
        imputation = MissImputation(["cat"], [], "test_imputation_category", True)

        result = imputation(data)

        self.assertEqual(result["cat"].iloc[2], "x")

    def test_zero_imputation(self):
        data = pd.DataFrame({"num": [1.0, np.nan, 5.0]})
        imputation = MissImputation([], ["num"], "test_imputation_zero", True, numeric_method="zero")

        result = imputation(data)

        self.assertEqual(result["num"].iloc[1], 0.0)


class TestColumnNormalize(PreprocessTestCase):

    def test_matches_pandas_statistics(self):
        data = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        normalize = ColumnNormalize(["a"], "test_normalize_values", True)

        result = normalize(data)

        expected = (data["a"] - data["a"].mean()) / data["a"].std()
        np.testing.assert_allclose(result["a"].to_numpy(), expected.to_numpy())

    def test_inference_uses_train_statistics(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        ColumnNormalize(["a"], "test_normalize_reuse", True)(train)

        test = pd.DataFrame({"a": [5.0]})
        result = ColumnNormalize(["a"], "test_normalize_reuse", False)(test)

        expected = (5.0 - train["a"].mean()) / train["a"].std()
        np.testing.assert_allclose(result["a"].to_numpy(), [expected])

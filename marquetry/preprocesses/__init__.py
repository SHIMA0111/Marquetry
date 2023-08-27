from typing import Literal

from marquetry.preprocesses.encodes.label_encode import LabelEncode
from marquetry.preprocesses.encodes.one_hot_encode import OneHotEncode

from marquetry.preprocesses.imputation.miss_imputation import MissImplementation

from marquetry.preprocesses.normalize.column_normalize import ColumnNormalize
from marquetry.preprocesses.normalize.column_standardize import ColumnStandardize


class ToEncodeData(object):
    def __init__(self, target_column, category_columns, numeric_columns, name,
                 imputation_category_method="mode", imputation_numeric_method="mean",
                 is_onehot=True, include_null=False,
                 normalize_method: Literal["normalize", "standardize"] = "normalize", allow_unknown_category=True):
        self._name = name
        self._target_column = target_column
        self._category_columns = category_columns
        self._numeric_columns = numeric_columns
        self._imputation_category_method = imputation_category_method
        self._imputation_numeric_method = imputation_numeric_method

        if is_onehot:
            self._category_encoder = OneHotEncode(
                self._category_columns, self._name, include_null, allow_unknown_category)
        else:
            unknown_handler: Literal["encode_specify_value", "raise_error"]
            if allow_unknown_category:
                unknown_handler = "encode_specify_value"
            else:
                unknown_handler = "raise_error"

            self._category_encoder = LabelEncode(self._category_columns, self._name,
                                                 include_null=include_null, treat_unknown=unknown_handler)

        if normalize_method == "normalize":
            self._norm_calculator = ColumnNormalize(self._numeric_columns, self._name)
        else:
            self._norm_calculator = ColumnStandardize(self._numeric_columns, self._name)

        self._imputation_runner = MissImplementation(self._category_columns, self._numeric_columns, self._name,
                                                     self._imputation_category_method, self._imputation_numeric_method)

    def __call__(self, data):
        data = self._norm_calculator(data)
        data = self._imputation_runner(data)
        data = self._category_encoder(data)

        return data

    def remove_old_statistic(self):
        self._norm_calculator.remove_old_statistic()
        self._imputation_runner.remove_old_statistic()
        self._category_encoder.remove_old_statistic()

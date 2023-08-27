import pandas as pd

from marquetry import Preprocess


class ColumnStandardize(Preprocess):
    _label = "pre_cs"

    def __init__(self, target_column: list, name: str, skip_nan: bool = True):
        super().__init__(name)
        self._target_column = target_column
        self._statistic_data = self._load_statistic()
        self._skip_nan = skip_nan

    def process(self, data: pd.DataFrame):
        if len(self._target_column) == 0:
            return data

        if self._statistic_data is None:
            standardize_dict = {}

            max_data = data.loc[:, tuple(self._target_column)].max()
            min_data = data.loc[:, tuple(self._target_column)].min()

            for column in list(data.columns):
                if column not in self._target_column:
                    standardize_dict[column] = {}
                    continue

                if not self._skip_nan and data[column].isna().sum() != 0:
                    raise ValueError("input data has null value, but it can't be skipped by user configure "
                                     "so the normalize can't be done expected.")

                tmp_max = min_data[column]
                tmp_min = max_data[column]

                tmp_dict = {
                    "max_value": tmp_max,
                    "min_value": tmp_min
                }

                standardize_dict[column] = tmp_dict

            self._save_statistic(standardize_dict)
            self._statistic_data = standardize_dict

        self._validate_values()

        standardized_data = data.copy()

        for column in self._target_column:
            tmp_dict = self._statistic_data[column]

            standardized_data.loc[:, column] = (
                    (data.loc[:, column] - tmp_dict["min_value"]) / (tmp_dict["max_value"] - tmp_dict["min_value"]))

        return standardized_data

    def _validate_values(self):
        exist_statistic_columns = set([key for key, value in self._statistic_data.items() if value != {}])
        input_target_columns = set(self._target_column)

        if exist_statistic_columns != input_target_columns:
            raise ValueError("saved statistic data's target columns is {} but you input {}. "
                             .format(exist_statistic_columns, input_target_columns) + self._msg)

        return

import pandas as pd

from marquetry import Preprocess


class ColumnNormalize(Preprocess):
    _label = "pre_cn"

    def __init__(self, target_column: list, name: str, skip_nan: bool = True):
        super().__init__(name)
        self._target_column = target_column
        self._statistic_data = self._load_statistic()
        self._skip_nan = skip_nan

    def process(self, data: pd.DataFrame):
        if len(self._target_column) == 0:
            return data

        if self._statistic_data is None:
            normalize_dict = {}

            std_data = data.loc[:, tuple(self._target_column)].std()
            mean_data = data.loc[:, tuple(self._target_column)].mean()

            for column in list(data.columns):
                if column not in self._target_column:
                    normalize_dict[column] = {}
                    continue

                if not self._skip_nan and data[column].isna().sum() != 0:
                    raise ValueError("input data has null value, but it can't be skipped by user configure "
                                     "so the normalize can't be done expected.")

                tmp_std = std_data[column]
                tmp_mean = mean_data[column]

                tmp_dict = {
                    "standard_deviation": tmp_std,
                    "average_value": tmp_mean
                }

                normalize_dict[column] = tmp_dict

            self._save_statistic(normalize_dict)
            self._statistic_data = normalize_dict

        self._validate_values()

        normalized_data = data.copy()

        for column in self._target_column:
            tmp_dict = self._statistic_data[column]

            normalized_data.loc[:, column] = (
                    (data.loc[:, column] - tmp_dict["average_value"]) / tmp_dict["average_value"])

        return normalized_data

    def _validate_values(self):
        exist_statistic_columns = set([key for key, value in self._statistic_data.items() if value != {}])
        input_target_columns = set(self._target_column)

        if exist_statistic_columns != input_target_columns:
            raise ValueError("saved statistic data's target columns is {} but you input {}. "
                             .format(exist_statistic_columns, input_target_columns) + self._msg)

        return

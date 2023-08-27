from typing import Literal

import numpy as np
import pandas as pd

from marquetry import Preprocess


class LabelEncode(Preprocess):
    _label = "pre_le"
    _msg = """if you use new data for the training, please use new `name` parameter or delete the old statistic data"""

    def __init__(self, target_column: list, name: str,
                 treat_unknown: Literal["no_label", "encode_specify_value", "raise_error"] = "encode_specify_value",
                 unknown_value: int = -1, include_null=False):
        super().__init__(name)
        self._target_column = target_column
        self._statistic_data = self._load_statistic()
        self._include_null = include_null
        self._treat_unknown = treat_unknown
        self._unknown_value = unknown_value

    def process(self, data: pd.DataFrame):
        if len(self._target_column) == 0:
            return data

        type_change_dict = {key: str for key in self._target_column}
        data = data.astype(type_change_dict).replace("nan", np.nan)

        if self._statistic_data is None:
            replace_dict = {}

            for column in list(data.columns):
                if column not in self._target_column:
                    replace_dict[column] = {}
                    continue

                tmp_series = data[column]
                unique_set = list(set(tmp_series))

                if not self._include_null:
                    unique_set = [unique_value for unique_value in unique_set if not pd.isna(unique_value)]

                class_nums = list(range(len(unique_set)))

                tmp_dict = dict(zip(unique_set, class_nums))
                replace_dict[column] = tmp_dict

            self._save_statistic(replace_dict)
            self._statistic_data = replace_dict

        unknown_handler = self._validate_values(data)

        if unknown_handler and self._treat_unknown == "encode_specify_value":
            for column in self._target_column:
                unique_set = set(data[column])
                statistic_unique = set(self._statistic_data[column].keys())

                unknown_set = unique_set - statistic_unique
                if len(unknown_set) == 0:
                    continue

                unknown_dict = {unknown_key: self._unknown_value for unknown_key in list(unknown_set)}

                self._statistic_data[column].update(unknown_dict)

        labeled_data = data.replace(self._statistic_data)

        return labeled_data

    def _validate_values(self, data):
        exist_statistic_columns = set([key for key, value in self._statistic_data.items() if value != {}])
        input_target_columns = set(self._target_column)

        diff_statistic_target = False

        if exist_statistic_columns != input_target_columns:
            raise ValueError("saved statistic data's target columns is {} but you input {}. "
                             .format(exist_statistic_columns, input_target_columns) + self._msg)

        for column in self._target_column:
            unique_set = set(data[column])
            if not self._include_null:
                unique_set = set(unique_value for unique_value in unique_set if not pd.isna(unique_value))

            statistic_set = set(pd.Series(self._statistic_data[column]).keys())

            diff_statistic_target = True if unique_set != statistic_set else False

            if unique_set != statistic_set and self._treat_unknown == "raise_error":
                raise ValueError("statistic data doesn't have {} category in '{}' but the input has it. "
                                 .format(",".join(sorted(list(unique_set - statistic_set))), column) + self._msg)

        return diff_statistic_target

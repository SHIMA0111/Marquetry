import numpy as np
import pandas as pd

from marquetry import Preprocess


class OneHotEncode(Preprocess):
    _label = "pre_ohe"

    def __init__(self, target_column: list, name: str, include_null=False, allow_unknown_value=False):
        super().__init__(name)
        self._target_column = target_column
        self._statistic_data = self._load_statistic()
        self._include_null = include_null
        self._unknown_value = allow_unknown_value

    def process(self, data: pd.DataFrame):
        if len(self._target_column) == 0:
            return data

        type_change_dict = {key: str for key in self._target_column}
        data = data.astype(type_change_dict).replace("nan", np.nan)

        batch_size = len(data)

        if self._statistic_data is None:
            one_hot_dict = {}
            for column in list(data.columns):
                if column not in self._target_column:
                    one_hot_dict[column] = {}
                    continue

                tmp_series = data[column]
                unique_set = list(set(tmp_series))

                if not self._include_null:
                    unique_set = [unique_value for unique_value in unique_set if not pd.isna(unique_value)]

                class_nums = list(range(len(unique_set)))

                tmp_dict = dict(zip(unique_set, class_nums))
                one_hot_dict[column] = tmp_dict

            self._save_statistic(one_hot_dict)
            self._statistic_data = one_hot_dict

        self._validate_values(data)

        replaced_data = data.replace(self._statistic_data)

        tmp_data = None
        for column in self._target_column:
            tmp_dict = self._statistic_data[column]
            unique_nums = len(tmp_dict)
            column_names = [column + "_" + str(i) for i in range(1, unique_nums)]

            unknown_replace_dict = {}
            if self._unknown_value:
                unique_set = set(data[column])
                statistic_set = set(pd.Series(tmp_dict).keys())
                unknown_values = list(unique_set - statistic_set)
                next_num = unique_nums
                unknown_replace_dict = {key: next_num for key in unknown_values}
                column_names.append(column + "_" + "unknown")
                unique_nums += 1

            one_hot_array = np.zeros((batch_size, unique_nums))
            one_hot_array[
                np.arange(batch_size), list(replaced_data[column].replace(unknown_replace_dict).astype(np.int32))] = 1.
            one_hot_df = pd.DataFrame(one_hot_array[:, 1:].astype(np.int32), columns=column_names)

            if tmp_data is None:
                tmp_data = one_hot_df
            else:
                tmp_data = pd.concat((tmp_data, one_hot_df), axis=1)

        encoded_data = replaced_data.drop(self._target_column, axis=1).reset_index(drop=True)
        encoded_data = pd.concat((encoded_data, tmp_data), axis=1)

        return encoded_data

    def _validate_values(self, data):
        exist_statistic_columns = set([key for key, value in self._statistic_data.items() if value != {}])
        input_target_columns = set(self._target_column)

        if exist_statistic_columns != input_target_columns:
            raise ValueError("saved statistic data's target columns is {} but you input {}. "
                             .format(exist_statistic_columns, input_target_columns) + self._msg)

        for column in self._target_column:
            unique_set = set(data[column])
            if not self._include_null:
                unique_set = set(unique_value for unique_value in unique_set if not pd.isna(unique_value))

            statistic_set = set(pd.Series(self._statistic_data[column]).keys())

            if unique_set != statistic_set and not self._unknown_value:
                raise ValueError("statistic data doesn't have {} category in '{}' but the input has it. "
                                 .format(",".join(sorted(list(unique_set - statistic_set))), column) + self._msg)

        return

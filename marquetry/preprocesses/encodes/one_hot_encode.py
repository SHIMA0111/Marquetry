import numpy as np
import pandas as pd

from marquetry import Preprocess


class OneHotEncode(Preprocess):
    """One-Hot Encoding for Categorical Data.

        OneHotEncode is a preprocessing step used to one-hot encode categorical columns in a Pandas DataFrame.
        This transformation converts categorical columns into a set of binary columns,
        where each unique category is represented as a separate binary column.

        Args:
            category_column (list): A list of column names to be one-hot encoded.
            name (str): The name of the OneHotEncode instance.
            is_train (bool): A boolean flag indicating whether this OneHotEncode instance is for training or inference.
            include_null (bool): Whether to include null or NaN values when calculating one-hot encoding.
                Defaults to False.
            allow_unknown_value (bool): Whether to allow unknown categories during one-hot encoding.
                If set to True, an additional binary column will be created for unknown categories.
                Defaults to False.

        Examples:
            >>> import pandas as pd
            >>> from marquetry.preprocesses import OneHotEncode
            >>> data = pd.DataFrame({'Category': ['A', 'B', 'A', 'C']})
            >>> one_hot_encoder = OneHotEncode(category_column=['Category'], name='one_hot_encoder', is_train=True)
            >>> encoded_data = one_hot_encoder(data)
            >>> print(encoded_data)
               Category_B  Category_C
            0           0           0
            1           1           0
            2           0           0
            3           0           1

        Note:
            OneHotEncode converts categorical columns into binary columns (0 and 1).
            And, You can control the inclusion of null or NaN values using the `include_null` parameter and,
            the `allow_unknown_value` parameter allows you to create an additional binary column for unknown categories.

            To avoid multicolinearity problem, remove the one binary category column.
    """

    _label = "pre_ohe"

    def __init__(self, category_column: list, name: str, is_train: bool, include_null=False, allow_unknown_value=False):
        super().__init__(name, is_train)
        self._category_column = category_column
        self._include_null = include_null
        self._unknown_value = allow_unknown_value

    def process(self, data: pd.DataFrame):
        """Process the input DataFrame by one-hot encoding specified columns.

            Args:
                data (:class:`pandas.DataFrame`): The input DataFrame to be one-hot encoded.

            Returns:
                pd.DataFrame: The one-hot encoded DataFrame.

            Caution:
                Generally, process will be called by ``Marquetry`` core.
                Please call like the ``EXAMPLES``.

        """

        if len(self._category_column) == 0:
            return data

        type_change_dict = {key: str for key in self._category_column}
        data = data.astype(type_change_dict).replace("nan", np.nan)

        batch_size = len(data)

        if self._statistic_data is None:
            self._calc_statistic(data)

        self._validate_values(data)

        replaced_data = data.replace(self._statistic_data)
        replaced_data = replaced_data[self._category_column]

        one_hot_data = None
        for column in self._category_column:
            tmp_dict = self._statistic_data[column]
            unique_nums = len(tmp_dict)
            column_names = [column + "_" + str(key) for i, key in enumerate(tmp_dict.keys()) if i != 0]

            # unknown value replacing process
            unknown_replace_dict = {}
            if self._unknown_value:
                unique_set = set(data[column])
                statistic_set = set(pd.Series(tmp_dict).keys())
                unknown_values = list(unique_set - statistic_set)
                next_num = unique_nums
                unknown_replace_dict = {key: next_num for key in unknown_values}
                column_names.append(column + "_" + "unknown")
                unique_nums += 1

            # create one-hot table
            one_hot_array = np.zeros((batch_size, unique_nums))
            one_hot_array[
                np.arange(batch_size), list(replaced_data[column].replace(unknown_replace_dict).astype(np.int32))] = 1.
            one_hot_df = pd.DataFrame(one_hot_array[:, 1:].astype(np.int32), columns=column_names)

            if one_hot_data is None:
                one_hot_data = one_hot_df
            else:
                one_hot_data = pd.concat((one_hot_data, one_hot_df), axis=1)

        return one_hot_data

    def _validate_values(self, data):
        exist_statistic_columns = set([key for key, value in self._statistic_data.items() if value != {}])
        input_target_columns = set(self._category_column)

        if exist_statistic_columns != input_target_columns:
            raise ValueError("saved statistic data's target columns is {} but you input {}. "
                             .format(exist_statistic_columns, input_target_columns) + self._msg)

        for column in self._category_column:
            unique_set = set(data[column])
            if not self._include_null:
                unique_set = set(unique_value for unique_value in unique_set if not pd.isna(unique_value))

            statistic_set = set(pd.Series(self._statistic_data[column]).keys())

            if unique_set != statistic_set and not self._unknown_value:
                raise ValueError("statistic data doesn't have {} category in '{}' but the input has it. "
                                 .format(",".join(sorted(list(unique_set - statistic_set))), column) + self._msg)

        return

    def _calc_statistic(self, data: pd.DataFrame):
        one_hot_dict = {}
        for column in list(data.columns):
            if column not in self._category_column:
                one_hot_dict[column] = {}
                continue

            tmp_series = data[column]
            unique_set = list(set(tmp_series))

            if not self._include_null:
                unique_set = [unique_value for unique_value in unique_set if not pd.isna(unique_value)]

            class_nums = list(range(len(unique_set)))

            tmp_dict = dict(zip(sorted(unique_set), class_nums))
            one_hot_dict[column] = tmp_dict

        self._save_statistic(one_hot_dict)
        self._statistic_data = one_hot_dict

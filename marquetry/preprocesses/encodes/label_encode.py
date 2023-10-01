from typing import Literal

import numpy as np
import pandas as pd

from marquetry import Preprocess


class LabelEncode(Preprocess):
    """Label Encoding for Categorical Data.

        LabelEncode is a preprocessing step used to encode categorical columns in a Pandas DataFrame into
        numerical labels.
        This is often done to prepare categorical data for machine learning models that require numerical input.

        Args:
            category_column (list): A list of column names to be label encoded.
            name (str): The name of the LabelEncode instance.
            is_train (bool): A boolean flag indicating whether this LabelEncode instance is for training or inference.
            specify_label (dict or None): A dictionary specifying custom labels for categories.
                The keys are column names, and the values are dictionaries mapping category values to label values.
                Defaults to None.
            treat_unknown (Literal["no_label", "encode_specify_value", "raise_error"]):
                How to handle unknown categories that are not found in the `specify_label` dictionary.

                Options are:

                - "no_label": Leave unknown categories unlabeled (default).
                - "encode_specify_value": Encode unknown categories with a specified label.
                - "raise_error": Raise an error if unknown categories are encountered.

            unknown_value (int): The label value to use for unknown categories
                when `treat_unknown` is set to "encode_specify_value".
                Defaults to -1.
            include_null (bool): Whether to include null or NaN values when calculating labels.
                Defaults to False.

        Examples:
            >>> import pandas as pd
            >>> from marquetry.preprocesses import LabelEncode
            >>> data = pd.DataFrame({'Category': ['A', 'B', 'A', 'C']})
            >>> label_encoder = LabelEncode(category_column=['Category'], name='label_encoder', is_train=True)
            >>> encoded_data = label_encoder(data)
            >>> print(encoded_data)
               Category
            0         0
            1         1
            2         0
            3         2

        Note:
            LabelEncode converts categorical columns into numerical labels.
            And, You can specify custom labels for categories using the `specify_label` parameter and,
            Unknown categories can be handled in different ways using the `treat_unknown` parameter.
    """

    _label = "pre_le"

    def __init__(self, category_column: list, name: str, is_train: bool, specify_label: dict = None,
                 treat_unknown: Literal["no_label", "encode_specify_value", "raise_error"] = "encode_specify_value",
                 unknown_value: int = -1, include_null=False):
        super().__init__(name, is_train)
        self._category_column = category_column
        self._include_null = include_null
        self._treat_unknown = treat_unknown
        self._unknown_value = unknown_value
        self._specify_relation_dict = specify_label if specify_label is not None else {}
        if not all(column in self._category_column for column in self._specify_relation_dict.keys()):
            raise ValueError("specify label include not exist label, this dataset has {}."
                             .format(self._category_column))

    def process(self, data: pd.DataFrame):
        """Process the input DataFrame by label encoding specified columns.

            Args:
                data (:class:`pandas.DataFrame`): The input DataFrame to be label encoded.

            Returns:
                pd.DataFrame: The label encoded DataFrame.

            Caution:
                Generally, process will be called by ``Marquetry`` core.
                Please call like the ``EXAMPLES``.

        """

        if len(self._category_column) == 0:
            return data

        type_change_dict = {key: str for key in self._category_column}
        data = data.astype(type_change_dict).replace("nan", np.nan)

        if self._statistic_data is None:
            self._calc_statistic(data)

        unknown_handler = self._validate_values(data)

        if unknown_handler and self._treat_unknown == "encode_specify_value":
            for column in self._category_column:
                unique_set = set(data[column])
                statistic_unique = set(self._statistic_data[column].keys())

                unknown_set = unique_set - statistic_unique
                if len(unknown_set) == 0:
                    continue

                unknown_dict = {unknown_key: self._unknown_value for unknown_key in list(unknown_set)}

                self._statistic_data[column].update(unknown_dict)

        labeled_data = data.replace(self._statistic_data)
        labeled_data = labeled_data[self._category_column]

        return labeled_data

    def _validate_values(self, data):
        exist_statistic_columns = set([key for key, value in self._statistic_data.items() if value != {}])
        input_target_columns = set(self._category_column)

        diff_statistic_target = False

        if exist_statistic_columns != input_target_columns:
            raise ValueError("saved statistic data's target columns is {} but you input {}. "
                             .format(exist_statistic_columns, input_target_columns) + self._msg)

        for column in self._category_column:
            unique_set = set(data[column])
            if not self._include_null:
                unique_set = set(unique_value for unique_value in unique_set if not pd.isna(unique_value))

            statistic_set = set(pd.Series(self._statistic_data[column]).keys())

            diff_statistic_target = True if unique_set != statistic_set else False

            if unique_set != statistic_set and self._treat_unknown == "raise_error":
                raise ValueError("statistic data doesn't have {} category in '{}' but the input has it. "
                                 .format(",".join(sorted(list(unique_set - statistic_set))), column) + self._msg)

        return diff_statistic_target

    def _calc_statistic(self, data: pd.DataFrame):
        replace_dict = {}

        for column in list(data.columns):
            if column not in self._category_column:
                replace_dict[column] = {}
                continue

            tmp_series = data[column]
            unique_set = set(tmp_series)

            if column in self._specify_relation_dict:
                tmp_dict = self._specify_relation_dict[column]

            else:
                if not self._include_null:
                    unique_set = [unique_value for unique_value in unique_set if not pd.isna(unique_value)]

                class_nums = list(range(len(unique_set)))

                tmp_dict = dict(zip(sorted(list(unique_set)), class_nums))

            if not len(set(unique_set) - set(tmp_dict.keys())) == 0:
                raise ValueError("{} has {} unique values but your specifying value is {}. "
                                 "If you want to specify label to this column, please cover all values."
                                 .format(column, len(unique_set), tmp_dict))

            replace_dict[column] = tmp_dict

        self._save_statistic(replace_dict)
        self._statistic_data = replace_dict

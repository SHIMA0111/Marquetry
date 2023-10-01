from typing import Literal

import pandas as pd

from marquetry.preprocesses.encodes.label_encode import LabelEncode
from marquetry.preprocesses.encodes.one_hot_encode import OneHotEncode

from marquetry.preprocesses.imputation.miss_imputation import MissImputation

from marquetry.preprocesses.normalize.column_normalize import ColumnNormalize
from marquetry.preprocesses.normalize.column_standardize import ColumnStandardize


class ToEncodeData(object):
    """Data Encoding and Preprocessing Utility.

    ToEncodeData is a utility class that facilitates data encoding and preprocessing for machine learning tasks.
    It combines several preprocessing steps such as label encoding, one-hot encoding, imputation, and normalization.

    Args:
        target_column (str): The name of the target column in the dataset.
        category_columns (list): A list of column names to be treated as categorical variables.
        numeric_columns (list): A list of column names to be treated as numeric variables.
        name (str): The name of the ToEncodeData instance.
        is_train (bool): A boolean flag indicating whether this instance is used for training or inference.
        imputation_category_method (str): The imputation method for categorical columns.
            Choices: "mode" (default) or "zero".
        imputation_numeric_method (str, optional): The imputation method for numeric columns.
            Choices: "mean" (default), "mode", "median", or "zero".
        label_encode_columns (list or None): A list of column names to be label encoded.
            Default is None which means no columns to label encoding.
        one_hot_encode_columns (list or None): A list of column names to be one-hot encoded.
            Default is None which means one-hot encode all category columns.
        include_null (bool): If True, include null values in encoding and imputation.
            Default is False.
        normalize_method (str): The method for normalizing columns.
            Choices: "normalize" (default), "standardize", or None (no normalization).
        specify_labels (dict or None): A dictionary specifying label mappings for label encoding.
            Default is None.
        allow_unknown_category (bool): If True, allow encoding of unknown categories.
            Default is True.

    Examples:
        >>> import pandas as pd
        >>> from marquetry.preprocesses import ToEncodeData
        >>> data = pd.DataFrame({'Category': ['A', 'B', 'A', 'C'], 'Numeric': [10, 20, 30, 40]})
        >>> encoder = ToEncodeData(target_column='Category', category_columns=['Category'], numeric_columns=['Numeric'],
        >>>                        name='encoder', is_train=True, allow_unknown_category=False)
        >>> encoded_data, unique_values = encoder(data)
        >>> print(encoded_data)
            Numeric  Category_B  Category_C
        0 -1.161895           0           0
        1 -0.387298           1           0
        2  0.387298           0           0
        3  1.161895           0           1
        >>> print(unique_values)
        {'Category': 3, 'Numeric': 'N/A'}

    Note:
        ToEncodeData combines label encoding, one-hot encoding, imputation, and normalization in one utility class.
        You can specify various encoding and imputation methods as well as which columns to encode.
        The `remove_old_statistic` method is provided to clear old statistics if needed.

    See Also:
        - :class:`LabelEncode`: Class for label encoding categorical columns.
        - :class:`OneHotEncode`: Class for one-hot encoding categorical columns.
        - :class:`MissImputation`: Class for imputing missing values in data.
        - :class:`ColumnNormalize`: Class for normalizing columns.
        - :class:`ColumnStandardize`: Class for standardizing columns.
    """

    def __init__(self,
                 target_column,
                 category_columns,
                 numeric_columns,
                 name,
                 is_train: bool,
                 imputation_category_method="mode",
                 imputation_numeric_method="mean",
                 label_encode_columns=None,
                 one_hot_encode_columns=None,
                 include_null=False,
                 normalize_method: Literal["normalize", "standardize", None] = "normalize",
                 specify_labels: dict = None,
                 allow_unknown_category=True):

        self._name = name
        self._target_column = target_column
        self._category_columns = category_columns
        self._numeric_columns = numeric_columns
        self._imputation_category_method = imputation_category_method
        self._imputation_numeric_method = imputation_numeric_method

        self._label_encode_columns = label_encode_columns if label_encode_columns is not None else []
        self._one_hot_encode_columns = \
            one_hot_encode_columns if one_hot_encode_columns is not None else self._category_columns

        self._one_hot_encoder = OneHotEncode(
            self._category_columns, self._name, is_train, include_null, allow_unknown_category)

        unknown_handler: Literal["encode_specify_value", "raise_error"]
        if allow_unknown_category:
            unknown_handler = "encode_specify_value"
        else:
            unknown_handler = "raise_error"

        self._label_encoder = LabelEncode(self._category_columns, self._name, is_train, specify_label=specify_labels,
                                          include_null=include_null, treat_unknown=unknown_handler)

        if normalize_method == "normalize":
            self._norm_calculator = (
                ColumnNormalize(self._numeric_columns + self._category_columns, self._name, is_train))
        elif normalize_method == "standardize":
            self._norm_calculator = (
                ColumnStandardize(self._numeric_columns + self._category_columns, self._name, is_train))
        else:
            self._norm_calculator = None

        self._imputation_runner = MissImputation(self._category_columns, self._numeric_columns, self._name, is_train,
                                                 self._imputation_category_method, self._imputation_numeric_method)

    def __call__(self, data):
        data = self._imputation_runner(data)
        labeled_table = self._label_encoder(data)
        one_hot_table = self._one_hot_encoder(data)

        unique_dict = {}
        for column_name in data.keys():
            if column_name in self._category_columns:
                unique_num = int(labeled_table[column_name].max()) + 1
                unique_dict[column_name] = unique_num
            else:
                unique_dict[column_name] = "N/A"

        labeled_table = labeled_table[self._category_columns]
        data = pd.concat((data.drop(self._category_columns, axis=1), labeled_table), axis=1)

        if self._norm_calculator is not None:
            data = self._norm_calculator(data)

        one_hot_columns = \
            [column for column in one_hot_table.columns if column.split("_")[0] in self._one_hot_encode_columns]
        one_hot_table = one_hot_table[one_hot_columns]

        output_table = pd.concat(
            (data.drop(self._one_hot_encode_columns, axis=1), one_hot_table), axis=1)

        return output_table, unique_dict

    def remove_old_statistic(self):
        """Remove old statistics used for imputation, normalization, and encoding."""
        if self._norm_calculator is not None:
            self._norm_calculator.remove_old_statistic()

        self._imputation_runner.remove_old_statistic()
        self._label_encoder.remove_old_statistic()
        self._one_hot_encoder.remove_old_statistic()

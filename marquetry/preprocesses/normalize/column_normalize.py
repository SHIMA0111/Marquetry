import pandas as pd

from marquetry import Preprocess


class ColumnNormalize(Preprocess):
    """Column Normalization for Specified Columns in a DataFrame.

        ColumnNormalize is a preprocessing step used to normalize specified columns in a Pandas DataFrame.
        It scales the values in the selected columns to have a mean of 0 and a standard deviation of 1.

        Args:
            norm_target_column (list): A list of column names to be normalized.
            name (str): The name of the ColumnNormalize instance.
            is_train (bool): A boolean flag indicating whether
                this ColumnNormalize instance is for training or inference.
            skip_nan (bool): If True, columns containing NaN (null) values will be skipped during normalization.
                If False, NaN values will raise an error.
                Defaults to True.

        Examples:
            >>> import pandas as pd
            >>> from marquetry.preprocesses import ColumnNormalize
            >>> data = pd.DataFrame({'FeatureA': [1.0, 2.0, 3.0, 4.0], 'FeatureB': [10.0, 20.0, 30.0, 40.0]})
            >>> normalizer = ColumnNormalize(norm_target_column=['FeatureA', 'FeatureB'], name='normalizer', is_train=True)
            >>> normalized_data = normalizer(data)
            >>> print(normalized_data)
               FeatureA  FeatureB
            0 -1.161895 -1.161895
            1 -0.387298 -0.387298
            2  0.387298  0.387298
            3  1.161895  1.161895

        Note:
            ColumnNormalize scales the values in the specified columns to have a mean of 0 and
            a standard deviation of 1.
            It provides an option to skip columns containing NaN (null) values during normalization.
    """

    _label = "pre_cn"

    def __init__(self, norm_target_column: list, name: str, is_train: bool, skip_nan: bool = True):
        super().__init__(name, is_train)
        self._norm_target_column = norm_target_column
        self._skip_nan = skip_nan

    def process(self, data: pd.DataFrame):
        """Process the input DataFrame by normalizing specified columns.

            Args:
                data (:class:`pandas.DataFrame`): The input DataFrame with columns to be normalized.

            Returns:
                pd.DataFrame: The DataFrame with specified columns normalized.

            Caution:
                Generally, process will be called by ``Marquetry`` core.
                Please call like the ``EXAMPLES``.

        """

        if len(self._norm_target_column) == 0:
            return data

        if self._statistic_data is None:
            self._calc_statistic(data)

        self._validate_values()
        normalized_data = data.copy()

        for column in self._norm_target_column:
            tmp_dict = self._statistic_data[column]

            normalized_data.loc[:, column] = (
                    (data.loc[:, column] - tmp_dict["average_value"]) / tmp_dict["standard_deviation"])

        return normalized_data

    def _validate_values(self):
        exist_statistic_columns = set([key for key, value in self._statistic_data.items() if value != {}])
        input_target_columns = set(self._norm_target_column)

        if exist_statistic_columns != input_target_columns:
            raise ValueError("saved statistic data's target columns is {} but you input {}. "
                             .format(exist_statistic_columns, input_target_columns) + self._msg)

        return

    def _calc_statistic(self, data: pd.DataFrame):
        normalize_dict = {}

        std_data = data.loc[:, tuple(self._norm_target_column)].std()
        mean_data = data.loc[:, tuple(self._norm_target_column)].mean()

        for column in list(data.columns):
            if column not in self._norm_target_column:
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

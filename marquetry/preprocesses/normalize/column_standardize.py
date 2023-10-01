import pandas as pd

from marquetry import Preprocess


class ColumnStandardize(Preprocess):
    """Column Standardization for Specified Columns in a DataFrame.

        ColumnStandardize is a preprocessing step used to standardize specified columns in a Pandas DataFrame.
        It scales the values in the selected columns to have a minimum of 0 and a maximum of 1.

        Args:
            std_target_column (list): A list of column names to be standardized.
            name (str): The name of the ColumnStandardize instance.
            is_train (bool): A boolean flag indicating whether
                this ColumnStandardize instance is for training or inference.
            skip_nan (bool): If True, columns containing NaN (null) values will be skipped during standardization.
                If False, NaN values will raise an error.
                Defaults to True.

        Examples:
            >>> import pandas as pd
            >>> from marquetry.preprocesses import ColumnStandardize
            >>> data = pd.DataFrame({'FeatureA': [1.0, 2.0, 3.0, 4.0], 'FeatureB': [10.0, 20.0, 30.0, 40.0]})
            >>> standardizer = ColumnStandardize(std_target_column=['FeatureA', 'FeatureB'], name='standardizer', is_train=True)
            >>> standardized_data = standardizer(data)
            >>> print(standardized_data)
               FeatureA  FeatureB
            0  1.000000  1.000000
            1  0.666667  0.666667
            2  0.333333  0.333333
            3 -0.000000 -0.000000

        Note:
            ColumnStandardize scales the values in the specified columns to have a minimum of 0 and a maximum of 1.
            It provides an option to skip columns containing NaN (null) values during standardization.
    """

    _label = "pre_cs"

    def __init__(self, std_target_column: list, name: str, is_train: bool, skip_nan: bool = True):
        super().__init__(name, is_train)
        self._std_target_column = std_target_column
        self._skip_nan = skip_nan

    def process(self, data: pd.DataFrame):
        """Process the input DataFrame by standardizing specified columns.

            Args:
                data (:class:`pandas.DataFrame`): The input DataFrame with columns to be standardized.

            Returns:
                pd.DataFrame: The DataFrame with specified columns standardized.

            Caution:
                Generally, process will be called by ``Marquetry`` core.
                Please call like the ``EXAMPLES``.
        """

        if len(self._std_target_column) == 0:
            return data

        if self._statistic_data is None:
            self._calc_statistic(data)

        self._validate_values()

        standardized_data = data.copy()

        for column in self._std_target_column:
            tmp_dict = self._statistic_data[column]

            standardized_data.loc[:, column] = (
                    (data.loc[:, column] - tmp_dict["min_value"]) / (tmp_dict["max_value"] - tmp_dict["min_value"]))

        return standardized_data

    def _validate_values(self):
        exist_statistic_columns = set([key for key, value in self._statistic_data.items() if value != {}])
        input_target_columns = set(self._std_target_column)

        if exist_statistic_columns != input_target_columns:
            raise ValueError("saved statistic data's target columns is {} but you input {}. "
                             .format(exist_statistic_columns, input_target_columns) + self._msg)

        return

    def _calc_statistic(self, data: pd.DataFrame):
        standardize_dict = {}

        max_data = data.loc[:, tuple(self._std_target_column)].max()
        min_data = data.loc[:, tuple(self._std_target_column)].min()

        for column in list(data.columns):
            if column not in self._std_target_column:
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

import pandas as pd

from marquetry import Preprocess


class MissImputation(Preprocess):
    """Missing Value Imputation for Categorical and Numeric Data.

        MissImputation is a preprocessing step used to impute missing values
        in both categorical and numeric columns of a Pandas DataFrame.
        It allows users to specify different imputation methods for each type of column.

        Args:
            category_column (list): A list of column names to be imputed with categorical values.
            numeric_column (list): A list of column names to be imputed with numeric values.
            name (str): The name of the MissImputation instance.
            is_train (bool): A boolean flag indicating whether
                this MissImputation instance is for training or inference.
            category_method (str): The imputation method to use for categorical columns.
                Supported methods are "mean" or "zero".
                Defaults to "mode".
            numeric_method (str): The imputation method to use for numeric columns.
                Supported methods are "mean", "mode", "median", and "zero".
                Defaults to "mean".

        Examples:
            >>> import pandas as pd
            >>> from marquetry.preprocesses import MissImputation
            >>> data = pd.DataFrame({'Category': ['A', 'B', 'A', None], 'Value': [1.0, None, 3.0, 4.0]})
            >>> imputer = MissImputation(category_column=['Category'], numeric_column=['Value'], name='imputer', is_train=True)
            >>> imputed_data = imputer(data)
            >>> print(imputed_data)
              Category     Value
            0        A  1.000000
            1        B  2.666667
            2        A  3.000000
            3        A  4.000000

        Note:
            MissImputation allows you to specify different imputation methods for categorical and numeric columns.
            Supported imputation methods are "mean", "mode", "median", and "zero".
            ("mean" and "median" supports only numerical columns.)
            Imputed values are calculated based on the specified method for each column.
            For categorical columns, "mode" is used by default, and for numeric columns, "mean" is used by default.
    """

    _label = "pre_mi"
    _enable_method = ("mean", "mode", "median", "zero")

    def __init__(self, category_column: list, numeric_column: list, name: str, is_train: bool,
                 category_method="mode", numeric_method="mean"):
        super().__init__(name, is_train)
        self._category_column = category_column
        self._numeric_column = numeric_column

        if category_method in self._enable_method and numeric_method in self._enable_method:
            self._category_method = category_method
            self._numeric_method = numeric_method
        else:
            enable_method_msg = "support method are {}.".format(",".join(self._enable_method))
            if category_method not in self._enable_method:
                raise TypeError(
                    "Category method: {} is not supported method. ".format(category_method) + enable_method_msg)
            elif numeric_method not in self._enable_method:
                raise TypeError(
                    "Numeric method: {} is not supported method. ".format(numeric_method) + enable_method_msg)
            else:
                raise TypeError(
                    "{} and {} are not supported. ".format(category_method, numeric_method) + enable_method_msg)

    def process(self, data: pd.DataFrame):
        """Process the input DataFrame by imputing missing values.

            Args:
                data (:class:`pandas.DataFrame`): The input DataFrame with missing values to be imputed.

            Returns:
                pd.DataFrame: The DataFrame with missing values imputed based on the specified methods.

            Caution:
                Generally, process will be called by ``Marquetry`` core.
                Please call like the ``EXAMPLES``.

        """

        if len(self._category_column + self._numeric_column) == 0:
            return data

        if self._statistic_data is None:
            self._calc_statistic(data)

        imputation_data = data.copy()

        for column in list(data.columns):
            if column in self._category_column:
                if pd.isna(self._statistic_data[column][self._category_method]):
                    raise TypeError("{} has no '{}' statistic due to the value can't convert Numeric value"
                                    .format(column, self._category_method))

                imputation_data.loc[:, column] = (
                    data.loc[:, column].fillna(self._statistic_data[column][self._category_method]))
            elif column in self._numeric_column:
                if pd.isna(self._statistic_data[column][self._numeric_method]):
                    raise TypeError("{} has no '{}' statistic due to the value can't convert Numeric value"
                                    .format(column, self._numeric_method))

                imputation_data.loc[:, column] = (
                    data.loc[:, column].fillna(self._statistic_data[column][self._numeric_method]))
            else:
                continue

        return imputation_data

    def _calc_statistic(self, data: pd.DataFrame):
        missing_imputation_dict = {}

        tmp_num_list = []
        tmp_str_dict = []

        for column in list(data.columns):
            if data[column].dtype in (int, float):
                tmp_num_list.append(column)
            else:
                tmp_str_dict.append(column)

        tmp_mean = data.loc[:, tmp_num_list].mean()
        tmp_median = data.loc[:, tmp_num_list].median()
        tmp_mode = data.mode()

        for column in list(data.columns):

            if column in tmp_str_dict:
                tmp_dict = {
                    "mean": None,
                    "median": None,
                    "mode": tmp_mode[column][0],
                    "zero": 0
                }
            else:
                tmp_dict = {
                    "mean": tmp_mean[column],
                    "median": tmp_median[column],
                    "mode": tmp_mode[column][0].astype(float),
                    "zero": 0
                }

            missing_imputation_dict[column] = tmp_dict

        self._save_statistic(missing_imputation_dict)
        self._statistic_data = missing_imputation_dict

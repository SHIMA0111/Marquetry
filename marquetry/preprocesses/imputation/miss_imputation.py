import pandas as pd

from marquetry import Preprocess


class MissImplementation(Preprocess):
    """A data preprocessing class for imputing missing values in a DataFrame.

        Args:
            category_column (list of strs): A list of column names with categorical values.
            numeric_column (list of strs): A list of column names with numeric values.
            name: A unique name for the missing value imputer.
            category_method: The method to impute missing values in categorical columns ("mode", "zero").
                Default is "mode".
            numeric_method: The method to impute missing values in numeric columns ("mean", "median", "zero").
                Default is "mean".

        Examples:
            >>> encoder = MissImplementation(["Embarked", ...], ["Fare", ...], "titanic")
            >>> labeled_data = encoder(data)

    """

    _label = "pre_mi"
    _enable_method = ("mean", "mode", "median", "zero")

    def __init__(self, category_column: list, numeric_column: list,
                 name: str, category_method="mode", numeric_method="mean"):
        super().__init__(name)
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

        """

        if len(self._category_column + self._numeric_column) == 0:
            return data

        if self._statistic_data is None:
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

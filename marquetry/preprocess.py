import numpy as np
import pandas as pd


# ===========================================================================
# data preprocess
# ===========================================================================
class LinearPreProcess(object):
    def __init__(self, categorical_columns: list, numerical_columns: list,
                 categorical_fill_method="mode", numerical_fill_method="median"):
        self.methods = ("mode", "median", "mean")
        if categorical_fill_method not in self.methods or numerical_fill_method not in self.methods:
            raise Exception("Fill method needs to be chosen from {}.".format(self.methods))

        self.categorical_fill_method = categorical_fill_method
        self.numerical_fill_method = numerical_fill_method

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.labeling_report = None
        self.normalize_report = None
        self.imputation_report = None
        self.one_hot_report = None

    def __call__(self, data: pd.DataFrame, to_normalize: bool = True,
                 to_one_hot: bool = True, fill_na: bool = True, is_train: bool = False, **additional_args):
        data.reset_index(inplace=True, drop=True)

        data = self.category_labeling(data, is_train=is_train)
        data = self.imputation_missing(data, is_train=is_train) if fill_na else data
        data = self.numerical_normalizing(data, is_train=is_train) if to_normalize else data
        data = self.one_hot_encoding(data, is_train=is_train) if to_one_hot else data

        return data

    def category_labeling(self, data: pd.DataFrame, is_train: bool = False):
        if len(self.categorical_columns) == 0:
            return data

        if is_train:
            replace_dict = {}

            for categorical_name in self.categorical_columns:
                tmp_series = data[categorical_name]
                unique_set = list(set(tmp_series))
                unique_set = [unique_value for unique_value in unique_set if not pd.isna(unique_value)]
                class_set = list(range(len(unique_set)))

                tmp_dict = dict(zip(unique_set, class_set))

                replace_dict[categorical_name] = tmp_dict

            self.labeling_report = replace_dict
            data = data.replace(self.labeling_report)
        else:
            if self.labeling_report is None:
                raise Exception("Preprocess test mode needs to be executed after train mode.")

            unknown_replace = {}
            for target_column in self.categorical_columns:
                unique_set = list(set(data[target_column]))
                unique_set = pd.Series(unique_set, dtype=str)

                next_num = len(self.labeling_report[target_column])
                train_exist_map = unique_set.isin(self.labeling_report[target_column].keys()).astype(float)
                train_exist_map -= 1.
                train_exist_map = train_exist_map.abs()
                train_exist_map *= next_num

                tmp_unknown = dict(zip(unique_set.to_dict().values(), train_exist_map.to_dict().values()))
                tmp_unknown = {key: value for key, value in tmp_unknown.items() if value != 0.0 and not pd.isna(key)}

                unknown_replace[target_column] = tmp_unknown

                if data[target_column].dtype in (int, float):
                    data[target_column] = data[target_column].astype(str)

            data = data.replace(unknown_replace)
            data = data.replace(self.labeling_report)

        return data

    def numerical_normalizing(self, data: pd.DataFrame, is_train: bool = False, axis=0):
        if len(self.numerical_columns) == 0:
            return data

        if is_train:
            data_mean = data.mean(axis=axis)
            data_std = data.std(axis=axis)

            norm_target_map = data.keys().isin(self.numerical_columns)
            data_mean.loc[~norm_target_map] = 0.
            data_std.loc[~norm_target_map] = 1.

            norm_report = list(zip(data_mean, data_std))
            norm_report = dict(zip(data.keys(), norm_report))

            self.normalize_report = norm_report

        else:
            if self.normalize_report is None:
                raise Exception("Preprocess test mode needs to be executed after train mode.")

            data_header = list(self.normalize_report.keys())
            data_mean = [statistic_tuple[0] for statistic_tuple in self.normalize_report.values()]
            data_std = [statistic_tuple[1] for statistic_tuple in self.normalize_report.values()]

            data_mean = pd.Series(data_mean, index=data_header)
            data_std = pd.Series(data_std, index=data_header)

        return (data - data_mean) / data_std

    def imputation_missing(self, data: pd.DataFrame, is_train: bool = False):
        if is_train:
            data_mode = data.mode(axis=0).iloc[0]
            data_mean = data.mean(axis=0)
            data_median = data.median(axis=0)

            imputation_report = {
                key: {"mode": mode, "mean": mean, "median": median}
                for key, mode, mean, median in zip(data.keys(), data_mode, data_mean, data_median)}

            self.imputation_report = imputation_report

        if self.imputation_report is None:
            raise Exception("Preprocess test mode needs to be executed after train mode.")

        missing_map = data.isna().sum()
        categorical_missing_map = list((missing_map != 0) & missing_map.keys().isin(self.categorical_columns))
        numerical_missing_map = list((missing_map != 0) & missing_map.keys().isin(self.numerical_columns))

        data_header = list(self.imputation_report.keys())
        categorical_imputation = [
            imputation_dict[self.categorical_fill_method] for imputation_dict in self.imputation_report.values()]
        numerical_imputation = [
            imputation_dict[self.numerical_fill_method] for imputation_dict in self.imputation_report.values()]

        categorical_imputation = pd.Series(categorical_imputation, index=data_header)
        numerical_imputation = pd.Series(numerical_imputation, index=data_header)

        data.fillna(categorical_imputation.loc[categorical_missing_map], inplace=True)
        data.fillna(numerical_imputation.loc[numerical_missing_map], inplace=True)

        return data

    def one_hot_encoding(self, data: pd.DataFrame, is_train: bool = False):
        if len(self.categorical_columns) == 0:
            return data

        if data.loc[:, self.categorical_columns].isna().values.sum() != 0:
            raise ValueError("The data has `NaN` so this data can't convert one-hot data.")

        one_hot_report = {}
        batch_size = len(data)
        for key in self.categorical_columns:
            if is_train:
                data_dim = len(set(data[key]))
                one_hot_report[key] = data_dim
            else:
                data_dim = self.one_hot_report[key]

            column_name = [key + "_" + str(i) for i in range(1, data_dim)]
            one_hot_array = np.zeros((batch_size, data_dim + 1))

            one_hot_array[np.arange(batch_size), list(data[key].astype(int))] = 1.
            one_hot_df = pd.DataFrame(one_hot_array[:, 1:-1], columns=column_name)

            data = pd.concat((data, one_hot_df), axis=1)

        self.one_hot_report = one_hot_report
        data = data.drop(self.categorical_columns, axis=1)

        return data

    def download_params(self):
        return {
            "label_report": self.labeling_report,
            "norm_report": self.normalize_report,
            "imputation_report": self.imputation_report,
            "one_hot_report": self.one_hot_report,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "categorical_fill_method": self.categorical_fill_method,
            "numerical_fill_method": self.numerical_fill_method,
        }

    def load_params(self, preprocess_params: dict, categorical_fill_method=None, numerical_fill_method=None):
        try:
            self.labeling_report = preprocess_params["label_report"]
            self.normalize_report = preprocess_params["norm_report"]
            self.imputation_report = preprocess_params["imputation_report"]
            self.one_hot_report = preprocess_params["one_hot_report"]
            self.categorical_columns = preprocess_params["categorical_columns"]
            self.numerical_columns = preprocess_params["numerical_columns"]

            self.categorical_fill_method = preprocess_params["categorical_fill_method"]
            self.numerical_fill_method = preprocess_params["numerical_fill_method"]

        except KeyError as e:
            raise KeyError("Your param file seems to be broken.")

        if categorical_fill_method is not None:
            if categorical_fill_method not in self.methods:
                print(
                    "{} is not supported so the default value {} will be used."
                    .format(categorical_fill_method, self.categorical_fill_method))
            else:
                self.categorical_fill_method = categorical_fill_method

        if numerical_fill_method is not None:
            if numerical_fill_method not in self.methods:
                print(
                    "{} is not supported so the default value {} will be used."
                    .format(numerical_fill_method, self.numerical_fill_method)
                )

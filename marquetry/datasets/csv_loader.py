import datetime
import json
import shutil
from typing import Literal
import os

import numpy as np
import pandas as pd

import marquetry
from marquetry import dataset
from marquetry.transformers import Compose, AsType


class CsvLoader(object):
    """Load CSV file as your custom dataset.

        CsvLoader loads your custom dataset formatted csv.

        If you want to use your original data as a train data, you can use this class.
        The data will be stored as your input `data_name`.

        Args:
            data_name (str): A name for your custom dataset.
            file_path (str): The file path to your CSV dataset.
            target_column (str): The name of the target column in the CSV dataset.
            category_columns (list): A list of column names that contain categorical data.
            numerical_columns (list): A list of column names that contain numerical data.
            target_path (str): The file path to a target CSV dataset (if separate from the main dataset).
            normalize_method (Literal["normalize", "standardize", None]):
                A method for normalizing or standardizing data. Default is "normalize".

        Examples:
            >>> loader = CsvLoader("custom_data", "data.csv", "target", ["category1", "category2"], ["num1", "num2"])
            ★ Please execute `load` method if you are okay the below configration ★
            #################################### config ####################################
            DataName: "custom_data"
            OriginalLocation: "data.csv" [FileSize: XXX KB]
            Target Column: "target"
            Category Columns: "category1, category2"
            Numerical Columns: "num1, num2"
            ################################################################################
            >>> loader.load()
            Complete import your csv data as "custom_data", please load your data via `CustomDataset`.

        Caution:
            The `data_name` needs to be a unique if you want to import the same name data,
            please execute :meth:`marquetry.datasets.delete_loaded_data`.

            And you can confirm the imported data by :meth:`marquetry.datasets.delete_loaded_data`.

    """

    _base_dir = os.path.join(marquetry.configuration.Config.CACHE_DIR, "csv_loader")

    def __init__(self, data_name: str, file_path: str, target_column: str, category_columns: list,
                 numerical_columns: list, target_path: str = None,
                 normalize_method: Literal["normalize", "standardize", None] = "normalize"):

        if target_column in category_columns or target_column in numerical_columns:
            violate_list = "category_columns" if target_column in category_columns else "numerical_columns"
            raise ValueError("`category_columns` and `numerical_columns` should not include `target_column`, "
                             "but {} includes in {}".format(target_column, violate_list))

        extension = file_path.rsplit(".")[-1]
        extension_target = target_path.rsplit(".")[-1] if target_path is not None else "csv"

        if extension != "csv" or extension_target != "csv":
            extension = extension if extension != "csv" else extension_target
            raise ValueError("CsvLoader support only csv file but got {}.".format(extension))

        self._display_path = file_path
        file_path = file_path.replace("~", os.path.expanduser("~"))
        target_path = target_path.replace("~", os.path.expanduser("~")) if target_path is not None else None
        if not os.path.exists(file_path):
            raise FileNotFoundError("\"{}\" doesn't exist in your computer, please check the path".format(file_path))

        if target_path is not None and not os.path.exists(target_path):
            raise FileNotFoundError("\"{}\" doesn't exist in your computer, please check the path".format(target_path))

        self._name = data_name
        self._file_path = file_path
        self._target_file_path = target_path

        self._target_column = target_column
        self._category_columns = category_columns
        self._numerical_columns = numerical_columns
        self._dropout_columns = []

        self._label_columns = []
        self._normalize_method = normalize_method

        self._filesize = None
        self._cache_dir = os.path.join(self._base_dir, data_name)
        if os.path.exists(self._cache_dir):
            raise ValueError("{} is already used, `data_name` should be unique.".format(data_name))

        print("\n★ Please execute `load` method if you are okay the below configration ★")
        print("#" * 36, "config", "#" * 36)
        print(self._print_config())
        print("#" * 80)

    def load(self):
        try:
            max_size = marquetry.configuration.Config.MAX_SIZE
            if self._filesize > max_size:
                raise RuntimeError("CsvLoader support up to {} csv, but your csv over the size. "
                                   "If you want to use more large file, please change `Config.MAX_SIZE`.\n"
                                   "The configuration may affects your process performance. Please caution."
                                   .format(self._get_size(max_size)))

            data = pd.read_csv(self._file_path)
            if self._target_file_path is not None:
                target = pd.read_csv(self._target_file_path)
                if self._target_column not in target:
                    raise RuntimeError("{} is not in the target csv file.".format(self._target_column))

                target = target[self._target_column]

                data = pd.concat((data, target), axis=1)

            else:
                if self._target_column not in data:
                    raise RuntimeError("{} is not in the data csv file. If you want to use other csv for the target, "
                                       "please specify the target file path via `target_path`".format(self._target_column))

            os.makedirs(self._cache_dir)
            file_path = os.path.join(self._cache_dir, "{}.parquet".format(self._name))
            data.to_parquet(file_path, index=False)

            with marquetry.using_config("CACHE_DIR", self._base_dir):
                preprocess = marquetry.preprocesses.ToEncodeData(
                    target_column=self._target_column,
                    category_columns=self._category_columns,
                    numeric_columns=self._numerical_columns,
                    name=self._name,
                    is_train=True,
                    label_encode_columns=self._label_columns,
                    one_hot_encode_columns=self._category_columns,
                    normalize_method=self._normalize_method
                )
                preprocess(data)

            self._save_file_profile()

            print("Complete import your csv data as \"{}\", please load your data via `CustomDataset`."
                  .format(self._name))

        except Exception as e:
            shutil.rmtree(self._cache_dir)
            raise e

    def _print_config(self):
        file_size = os.path.getsize(self._file_path)
        self._filesize = file_size

        content = ""
        content += ("DataName: \"{}\" \n"
                    "OriginalLocation: \"{}\" [FileSize: {}]\n"
                    .format(self._name, self._display_path, self._get_size(file_size)))
        if self._target_file_path is not None:
            file_size = os.path.getsize(self._target_file_path)
            content += ("TargetOriginLocation: \"{}\" [FileSize: {}]\n"
                        .format(self._target_file_path, self._get_size(file_size)))
        content += "Target Column: \"{}\"\n".format(self._target_column)
        content += "Category Columns: \"{}\"\n".format(", ".join(self._category_columns))
        content += "Numerical Columns: \"{}\"\n".format(", ".join(self._numerical_columns))

        return content

    @staticmethod
    def _get_size(file_size):
        units = ("B", "KB", "MB", "GB")
        size_unit = 0
        while not (file_size < 1024. or size_unit > 3):
            size_unit += 1
            file_size /= 1024.

        return "{:.2f} {}".format(file_size, units[size_unit])

    def _save_file_profile(self):
        profile = {
            "category_columns": self._category_columns,
            "numerical_columns": self._numerical_columns,
            "target_column": self._target_column,
            "normalize_method": self._normalize_method,
            "created_time": str(datetime.datetime.now()),
        }

        file_path = os.path.join(self._cache_dir, "profile.json")

        with open(file_path, "w") as f:
            json.dump(profile, f)

        return


class CustomDataset(dataset.Dataset):
    """Custom dataset class for loading and preprocessing data from loaded data by `CsvLoader`.

        This class is designed to work with custom datasets loaded from `CsvLoader`.
        It provides various preprocessing options,
        such as label encoding, one-hot encoding, imputation, and data type conversion.

        Args:
            data_name (str): A name for your custom dataset set in `CsvLoader`.
            label_columns (list): A list of column names to be label encoded.
                Default is None.
            drop_columns (list): A list of column names to be dropped from the dataset.
                Default is None.
            imputation_category_method (Literal["mode", "zero"]):
                Method for imputing missing values in categorical columns. Default is "mode".
            imputation_numerical_method (Literal["mean", "median", "mode", "zero"]):
                Method for imputing missing values in numerical columns. Default is "median".
            specify_labels (dict): A dictionary specifying labels for categorical variables.
                Default is None.
            include_null (bool): Whether to include null values treat as one of category value.
                Default is False.
            allow_unknown_category (bool): Whether to allow unknown categories during label encoding.
                Default is True.
            pre_process (bool): Whether to perform data preprocessing.
                Default is True.
            down_float_bit (bool): Whether to downcast data types to float32 after preprocessing.
                Default is True.

        Examples:
            >>> dataset = CustomDataset(data_name="custom_data")
            >>> test_dataset = dataset.test_data(test_csv_path="/path/to/test/data.csv", output_type="numpy")

        Caution:
            When you want to get test data from your custom csv corresponding the loaded data, you can get it by
            `test_data` method.

            From the method, you can get a data performed the same preprocess as the train data to test data you input.

        Error:
            The test data must have the same construction("column name" and "column order") as the train data
            exept for the target column.

    """

    _base_dir = os.path.join(marquetry.configuration.Config.CACHE_DIR, "csv_loader")

    def __init__(self, data_name: str, label_columns: list = None, drop_columns: list = None,
                 imputation_category_method: Literal["mode", "zero"] = "mode",
                 imputation_numerical_method: Literal["mean", "median", "mode", "zero"] = "median",
                 specify_labels: dict = None, include_null: bool = False, allow_unknown_category: bool = True,
                 pre_process: bool = True, down_float_bit: bool = True):
        self._name = data_name
        self._cache_dir = os.path.join(self._base_dir, self._name)
        self._category_columns = []
        self._numerical_columns = []
        self._target_column = None
        self._normalize_method = None
        self._created_time = None

        self._load_profile()

        label_columns = label_columns if label_columns is not None else []
        if len(label_columns) == 0:
            one_hot_columns = self._category_columns
        else:
            one_hot_columns = [column for column in self._category_columns if column not in label_columns]

        self._label_columns = label_columns
        self._one_hot_columns = one_hot_columns
        self._imputation_category_method = imputation_category_method
        self._imputation_numerical_method = imputation_numerical_method
        self._specify_labels = specify_labels
        self._include_null = include_null
        self._allow_unknown_category = allow_unknown_category

        self._preprocess = pre_process
        self._dropout_columns = drop_columns if drop_columns is not None else []
        self._down_float_bit = down_float_bit

        self._source_columns = None
        self._unique_dict = None

        if down_float_bit and pre_process:
            transform = Compose([AsType(dtype=np.float32)])
        else:
            transform = None

        super().__init__(train=True, transform=transform, target_transform=None)

    def _set_data(self, test_data: pd.DataFrame = None):
        if test_data is not None:
            data = test_data
        else:
            file_name = "{}.parquet".format(self._name)
            file_path = os.path.join(self._cache_dir, file_name)
            data = pd.read_parquet(file_path)

        if self._preprocess:
            with marquetry.using_config("CACHE_DIR", self._cache_dir):
                preprocess = marquetry.preprocesses.ToEncodeData(
                    target_column=self._target_column, category_columns=self._category_columns,
                    numeric_columns=self._numerical_columns, name=self._name, is_train=self.train,
                    imputation_category_method=self._imputation_category_method,
                    imputation_numeric_method=self._imputation_numerical_method,
                    label_encode_columns=self._label_columns, one_hot_encode_columns=self._one_hot_columns,
                    include_null=self._include_null, normalize_method=self._normalize_method,
                    specify_labels=self._specify_labels, allow_unknown_category=self._allow_unknown_category
                )

                encoded_data, unique_dict = preprocess(data)

        for dropout_column in self._dropout_columns:
            drop_regex = encoded_data.filter(regex=dropout_column)
            if drop_regex.shape[1] != 0:
                encoded_data.drop(drop_regex.columns, axis=1, inplace=True)

        source = encoded_data.drop(self._target_column, axis=1)
        if test_data is None:
            self._source_columns = list(source.columns)
            self._unique_dict = unique_dict

            self.source = source.to_numpy()
            self.target = encoded_data.loc[:, self._target_column].astype(int).to_numpy()

        else:
            return source

    def test_data(self, test_csv_path: str, output_type: Literal["numpy", "pandas"] = "numpy"):
        test_csv_path = test_csv_path.replace("~", os.path.expanduser("~"))
        if not os.path.exists(test_csv_path):
            raise FileNotFoundError("\"{}\" doesn't exist in your computer, please check the path"
                                    .format(test_csv_path))

        extension = test_csv_path.rsplit(".")[-1]
        if extension != "csv":
            raise ValueError("`test_data` of `CustomDataset` support only csv file but got {}.".format(extension))

        max_size = marquetry.configuration.Config.MAX_SIZE
        file_size = os.path.getsize(test_csv_path)

        if file_size > max_size:
            raise RuntimeError("Marquetry support up to {} csv, but your csv over the size. "
                               "If you want to use more large file, please change `Config.MAX_SIZE`.\n"
                               "The configuration may affects your process performance. Please caution."
                               .format(self._get_size(max_size)))

        data = pd.read_csv(test_csv_path)
        data[self._target_column] = 0

        source = self._set_data(data)

        if output_type == "numpy":
            source = source.to_numpy()

        return source

    def _load_profile(self):
        if not os.path.exists(self._cache_dir):
            raise ValueError("data: {} doesn't exist on your machine, please execute `CsvLoader` at first."
                             .format(self._name))

        file_path = os.path.join(self._cache_dir, "profile.json")

        with open(file_path, "r") as f:
            profile_data = json.load(f)

        self._category_columns = profile_data["category_columns"]
        self._numerical_columns = profile_data["numerical_columns"]
        self._target_column = profile_data["target_column"]
        self._normalize_method = profile_data["normalize_method"]

    @staticmethod
    def _get_size(file_size):
        units = ("B", "KB", "MB", "GB")
        size_unit = 0
        while not (file_size < 1024. or size_unit > 3):
            size_unit += 1
            file_size /= 1024.

        return "{:.2f} {}".format(file_size, units[size_unit])


def delete_loaded_data(data_name: str):
    """Delete a previously loaded custom dataset and its associated cached files.

        Args:
            data_name (str): The name of the custom dataset to delete.

        Warnings:
            This function permanently deletes the dataset and its cached files. Make sure you want to proceed.

        Examples:
            >>> delete_loaded_data("my_custom_dataset")
    """

    base_dir = os.path.join(marquetry.configuration.Config.CACHE_DIR, "csv_loader")
    cache_dir = os.path.join(base_dir, data_name)

    if not os.path.exists(cache_dir):
        raise ValueError("{} doesn't exist on your machine.".format(data_name))

    check = input("Does okay to delete \"{}\"? you answer this by except for (\"y\" or \"yes\"), "
                  "the process will be abort."
                  .format(data_name))

    if check.lower() in ("y", "yes"):
        shutil.rmtree(cache_dir)
    else:
        print("Abort deletion.")


def loaded_list():
    """List the names of loaded custom datasets.

        Lists the names of loaded custom datasets that have been cached for use.
        You can use these dataset names to access the cached data.

        Examples:
            >>> loaded_list()
    """

    base_dir = os.path.join(marquetry.configuration.Config.CACHE_DIR, "csv_loader")

    dir_detail = os.listdir(base_dir)
    name_list = [name for name in dir_detail if os.path.isdir(os.path.join(base_dir, name))]

    name_list_str = ""
    col_num = 0
    for i, name in enumerate(name_list):
        if len(name_list_str) > 70 or (len(name_list_str) != 0 and len(name) > 50):
            name_list_str += "\n"
            col_num = 0

        if col_num == 0:
            name_list_str += " "

        if len(name_list) <= 1 or i + 1 == len(name_list):
            name_list_str += name
            col_num += 1
        else:
            name_list_str += name + ", "
            col_num += 1

    print("#" * 26, "Loaded data name", "#" * 26)
    print(name_list_str)
    print("#" * 70)

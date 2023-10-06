import os

import numpy as np
import pandas as pd

from marquetry import dataset, preprocesses
from marquetry.transformers import compose, astypes
from marquetry.utils import get_file


class Titanic(dataset.Dataset):
    """Get the Titanic dataset.

        Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.

        The sinking of the Titanic is one of the most infamous shipwrecks in history.

        On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding
        with an iceberg. Unfortunately, there weren't enough lifeboats for everyone onboard,
        resulting in the death of 1502 out of 2224 passengers and crew.
        While there was some element of luck involved in surviving,
        it seems some groups of people were more likely to survive than others.
        In this challenge, we ask you to build a predictive model that answers the question:
        “what sorts of people were more likely to survive?”
        using passenger data (ie name, age, gender, socio-economic class, etc).
        (From kaggle competition description.)

        Args:
            train (bool): If True, loads the training dataset.
                Default is True. (When you want to get test data, you should use :meth:`test_data`.)
            train_rate (float, optional): The fraction of data to use for training when `train` is True.
                Should be in the range (0.0, 1.0). Default is 0.8.
            label_columns (list): List of column names to be treated as labels.
                Default is None.
            drop_columns (list): List of column names to be dropped from the dataset.
                Default is None.
            pre_process (bool): If True, preprocesses the dataset by encoding, imputing, and normalizing columns.
                Default is True.
            down_float_bit (bool): If True, downcasts float data to float32 for memory efficiency.
                Default is True.

        Examples:
            >>> from marquetry.datasets import Titanic
            >>> train_dataset = Titanic(train=True, train_rate=0.8, label_columns=["sex"], pre_process=True)
            >>> test_dataset = train_dataset.test_data()
            >>> print(len(train_dataset), len(test_dataset))
            1047 262
            >>> print(train_dataset[0])
            (array([-0.17157276,  0.4409694, -0.44408074, ...,  0., 0.,  0.], dtype=float32), 0)
            >>> print(train_dataset.source_columns)
            ['age', 'sibsp', 'parch', 'fare', 'sex', 'pclass_2', 'pclass_3', ..., 'cabin_C82', 'ticket_unknown']
            >>> print(train_dataset.target_columns)
            survived
            >>> print(train_dataset.unique_numbers)
            {'pclass': 3, 'survived': 'N/A', ..., 'fare': 'N/A', 'cabin': 157, 'embarked': 3}

        Note:
            The Titanic dataset is commonly used for binary classification tasks.
            It contains information about passengers on the Titanic, including their features and whether they survived.

        Caution:
            When you want to get test data, we suggest to use :meth:`test_data` instead of the ``train=False``
            due to Titanic dataset needs to be set the same setting as the train data when you get test data.

            If the setting is changed, the test data can't get correctly.
            From :meth:`test_data`, you can get the test dataset correctly.

    """

    def __init__(self, train=True, train_rate=0.8, label_columns: list = None,
                 drop_columns: list = None, pre_process: bool = True, down_float_bit: bool = True):

        if not 0.0 <= train_rate <= 1.0:
            raise ValueError("train_rate should in the range of (0.0, 1.0), but got {}".format(train_rate))

        self._category_columns = ["pclass", "sex", "cabin", "embarked", "name", "ticket"]
        self._numerical_columns = ["age", "sibsp", "parch", "fare"]
        self._target_column = "survived"

        label_columns = label_columns if label_columns is not None else []
        if len(label_columns) == 0:
            one_hot_columns = self._category_columns
        else:
            one_hot_columns = [column for column in self._category_columns if column not in label_columns]

        self._label_columns = label_columns
        self._one_hot_columns = one_hot_columns

        self.train_rate = train_rate

        self.pre_process = pre_process
        self.drop_columns = drop_columns if drop_columns is not None else []

        self._source_columns = None
        self._unique_dict = None

        self._down_float_bit = down_float_bit
        if down_float_bit and pre_process:
            transform = compose.Compose([astypes.AsType(dtype=np.float32)])
        else:
            transform = None

        super().__init__(train, transform, None)

    def _set_data(self, **kwargs):
        url = "https://biostat.app.vumc.org/wiki/pub/Main/DataSets/titanic3.csv"

        data_path = get_file(url)
        data = self._load_data(data_path)

        source = data.drop("survived", axis=1)
        self._source_columns = list(source.columns)

        self.source = source.to_numpy()
        self.target = data.loc[:, "survived"].astype(int).to_numpy()

    def _load_data(self, file_path):
        titanic_df = pd.read_csv(file_path)
        titanic_df = self._remove_leak_column(titanic_df, file_path)

        train_last_index = int(len(titanic_df) * self.train_rate)
        if self.train:
            titanic_df = titanic_df.iloc[:train_last_index, :]
        else:
            titanic_df = titanic_df.iloc[train_last_index:, :]

        if self.pre_process:
            preprocess = preprocesses.ToEncodeData(
                target_column="survived", category_columns=self._category_columns,
                numeric_columns=self._numerical_columns, name="titanic_dataset", is_train=self.train,
                imputation_category_method="mode", imputation_numeric_method="median",
                label_encode_columns=self._label_columns, one_hot_encode_columns=self._one_hot_columns,
                normalize_method="normalize"
            )

            titanic_df, self._unique_dict = preprocess(titanic_df)

        for drop_column in self.drop_columns:
            drop_regex = titanic_df.filter(regex=drop_column)
            if drop_regex.shape[1] != 0:
                titanic_df = titanic_df.drop(drop_regex.columns, axis=1)

        return titanic_df

    @staticmethod
    def _remove_leak_column(data: pd.DataFrame, file_path):
        leak_columns = ("body", "home.dest", "boat")

        original_columns = data.columns
        for leak_column in leak_columns:
            if leak_column in data:
                data = data.drop(leak_column, axis=1)

        if any(column in original_columns for column in leak_columns):
            data = data.sample(frac=1, random_state=2023)
            data.reset_index(drop=True, inplace=True)
            data.to_csv(file_path, index=False)

        return data

    def test_data(self):
        test_dataset = Titanic(train=False, train_rate=self.train_rate, label_columns=self._label_columns,
                               drop_columns=self.drop_columns, pre_process=self.pre_process,
                               down_float_bit=self._down_float_bit)
        test_dataset._unique_dict = "Test data can't calculate the unique nums."

        return test_dataset

    @property
    def source_columns(self):
        return self._source_columns

    @property
    def target_columns(self):
        return self._target_column

    @property
    def unique_numbers(self):
        return self._unique_dict

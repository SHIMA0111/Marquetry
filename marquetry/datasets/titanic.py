import os

import pandas as pd

from marquetry import dataset, preprocesses, transformers
from marquetry.utils import get_file


class Titanic(dataset.Dataset):
    """Get the Titanic dataset.

        Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.

        The sinking of the Titanic is one of the most infamous shipwrecks in history.

        On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with
        an iceberg. Unfortunately, there weren't enough lifeboats for everyone onboard,
        resulting in the death of 1502 out of 2224 passengers and crew.
        While there was some element of luck involved in surviving,
        it seems some groups of people were more likely to survive than others.
        In this challenge, we ask you to build a predictive model that answers the question:
        “what sorts of people were more likely to survive?”
        using passenger data (ie name, age, gender, socio-economic class, etc).
        (From kaggle competition description.)

    """
    def __init__(self, train=True, transform=transformers.ToFloat(), target_transform=None,
                 train_rate=0.8, is_one_hot=True, **kwargs):
        self.train_rate = train_rate
        self.is_one_hot = is_one_hot

        self.target_columns = None
        self.source_columns = None
        self.drop_columns = kwargs.get("drop_columns", [])
        self.remove_old_statistic = kwargs.get("remove_old_statistic", False)

        if self.remove_old_statistic and not train:
            raise Exception("test data need to be transformed by the train statistic "
                            "so you can't delete statistic data in test mode.")

        super().__init__(train, transform, target_transform, **kwargs)

    def _set_data(self, **kwargs):
        url = "https://biostat.app.vumc.org/wiki/pub/Main/DataSets/titanic3.csv"

        data_path = get_file(url)
        data = self._load_data(data_path, **kwargs)

        source = data.drop("survived", axis=1)
        target = data.loc[:, ["index", "survived"]].astype(int)
        self.target_columns = list(target.drop("index", axis=1).keys())
        self.source_columns = list(source.drop("index", axis=1).keys())

        source = source.to_numpy()
        target = target.to_numpy()

        self.target = target[:, 1:]
        self.source = source[:, 1:]

    def _load_data(self, file_path, **kwargs):
        titanic_df = pd.read_csv(file_path)

        change_flg = False
        if "body" in titanic_df:
            # "body" means body identity number, this put on only dead peoples.
            titanic_df = titanic_df.drop("body", axis=1)
            change_flg = True
        if "home.dest" in titanic_df:
            titanic_df = titanic_df.drop("home.dest", axis=1)
            change_flg = True
        if "boat" in titanic_df:
            # "boat" means lifeboat identifier, almost people having this data survive.
            titanic_df = titanic_df.drop("boat", axis=1)
            change_flg = True

        if self.train:
            param_path = file_path[:file_path.rfind(".")] + ".json"
            if os.path.exists(param_path):
                os.remove(param_path)
            if change_flg:
                titanic_df = titanic_df.sample(frac=1, random_state=2023)
                titanic_df.reset_index(inplace=True, drop=True)
                titanic_df.to_csv(file_path, index=False)

        for drop_column in self.drop_columns:
            if drop_column in titanic_df:
                titanic_df = titanic_df.drop(drop_column, axis=1)

        train_last_index = int(len(titanic_df) * self.train_rate)
        if self.train:
            titanic_df = titanic_df.iloc[:train_last_index, :]
        else:
            titanic_df = titanic_df.iloc[train_last_index:, :]

        categorical_columns = [
            categorical_name for categorical_name in self.categorical_columns
            if categorical_name in list(titanic_df.keys())
        ]

        numerical_columns = [
            numerical_name for numerical_name in self.numerical_columns
            if numerical_name in list(titanic_df.keys())
        ]

        preprocess = preprocesses.ToEncodeData(
            target_column="survived", category_columns=categorical_columns, numeric_columns=numerical_columns,
            name="titanic_dataset", imputation_category_method="mode", imputation_numeric_method="median",
            is_onehot=self.is_one_hot, normalize_method="standardize"
        )

        if self.remove_old_statistic:
            preprocess.remove_old_statistic()

        try:
            titanic_df = preprocess(titanic_df)

        except ValueError as e:
            raise ValueError("statistic data unmatch your input data if you want to use new data, "
                             "please specify `remove_old_statistic` as True in train mode.")

        index_series = pd.Series(titanic_df.index, name="index")
        titanic_df = pd.concat([index_series, titanic_df], axis=1)

        return titanic_df

    @property
    def categorical_columns(self):
        return ["pclass", "sex", "cabin", "embarked", "name", "ticket"]

    @property
    def numerical_columns(self):
        return ["age", "sibsp", "parch", "fare"]

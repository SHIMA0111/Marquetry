import gzip

import numpy as np
import pandas as pd

from marquetry.utils import get_file, label_encoder, data_normalizer, fill_missing
from marquetry.transformers import Compose, Flatten, ToFloat, Normalize


# ===========================================================================
# Dataset base class
# ===========================================================================
class Dataset(object):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.transform is None:
            self.transform = lambda x: x

        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.source = None
        self.target = None

        self._set_data()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.target is None:
            return self.transform(self.source[index]), None
        else:
            return self.transform(self.source[index]), self.target_transform(self.target[index])

    def __len__(self):
        return len(self.source)

    def _set_data(self, *args):
        raise NotImplementedError()


# ===========================================================================
# MNIST / Titanic / SpaceShipTitanic
# ===========================================================================
class MNIST(Dataset):
    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(), Normalize(0., 255)]), target_transform=None):
        super().__init__(train, transform, target_transform)

    def _set_data(self):
        url = "http://yann.lecun.com/exdb/mnist/"

        train_files = {
            "source": "train-images-idx3-ubyte.gz",
            "target": "train-labels-idx1-ubyte.gz"
        }
        test_files = {
            "source": "t10k-images-idx3-ubyte.gz",
            "target": "t10k-labels-idx1-ubyte.gz"
        }

        files = train_files if self.train else test_files
        source_path = get_file(url + files["source"])
        target_path = get_file(url + files["target"])

        self.source = self._load_source(source_path)
        self.target = self._load_target(target_path)

    @staticmethod
    def _load_source(file_path):
        with gzip.open(file_path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        source = data.reshape((-1, 1, 28, 28))

        return source

    @staticmethod
    def _load_target(file_path):
        with gzip.open(file_path, "rb") as f:
            target = np.frombuffer(f.read(), np.uint8, offset=8)

        return target

    @property
    def labels(self):
        return {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}


class Titanic(Dataset):
    """
    Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.
    """
    def __init__(self, train=True, transform=ToFloat(), target_transform=None,
                 train_rate=0.8, is_raw=False, auto_fillna=True):
        self.train_rate = train_rate
        self.is_raw = is_raw
        self.auto_fillna = auto_fillna
        self.columns = None
        super().__init__(train, transform, target_transform)

    def _set_data(self):
        url = "https://biostat.app.vumc.org/wiki/pub/Main/DataSets/titanic3.csv"

        data_path = get_file(url)
        data, encode_report, norm_report, fill_report = self._load_data(data_path, self.is_raw)
        self.columns = list(data.columns)
        train_last_index = int(len(data) * self.train_rate)

        source = data.drop("survived", axis=1)
        target = data.loc[:, ["index", "survived"]].astype(int)

        source = source.to_numpy()
        target = target.to_numpy()

        if self.train:
            self.target = target[:train_last_index, 1:]
            self.source = source[:train_last_index, 1:]
        else:
            self.target = target[train_last_index:, 1:]
            self.source = source[train_last_index:, 1:]

    def _load_data(self, file_path, is_raw):
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


        if change_flg:
            titanic_df.reset_index(inplace=True)
            titanic_df.to_csv(file_path, index=False)

        encode_report, norm_report, fill_report = None, None, None
        if not is_raw:
            titanic_df, encode_report = label_encoder(titanic_df, self.categorical_columns)
            titanic_df, norm_report = data_normalizer(titanic_df, self.numerical_columns)

        if self.auto_fillna:
            titanic_df, fill_report = fill_missing(titanic_df, self.categorical_columns, self.numerical_columns)

        return titanic_df, encode_report, norm_report, fill_report

    @property
    def categorical_columns(self):
        return ["pclass", "sex", "cabin", "embarked", "name", "ticket"]

    @property
    def numerical_columns(self):
        return ["age", "sibsp", "parch", "fare"]

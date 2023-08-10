import gzip
import os.path

import numpy as np
import pandas as pd

import matrixml
from matrixml.utils import get_file


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
    def __init__(self, train=True, transform=None, target_transform=None, train_rate=0.8, is_raw=False):
        self.train_rate = train_rate
        self.is_raw = is_raw
        super().__init__(train, transform, target_transform)

    def _set_data(self):
        url = "https://biostat.app.vumc.org/wiki/pub/Main/DataSets/titanic3.csv"

        data_path = get_file(url)
        data = self._load_data(data_path, self.is_raw)
        train_last_index = int(len(data) * self.train_rate)

        source = data.drop("survived", axis=1)
        target = data.loc[:, ["index", "survived"]]
        if self.train:
            self.target = target[:train_last_index]
            self.source = source[:train_last_index]
        else:
            self.target = target[train_last_index:]
            self.source = source[train_last_index:]


    @staticmethod
    def _load_data(file_path, is_raw):
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

        if not is_raw:
            pass

        return titanic_df

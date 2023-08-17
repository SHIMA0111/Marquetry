import gzip
import os.path
import json

import numpy as np
import pandas as pd

from marquetry.utils import get_file
from marquetry.preprocess import LinearPreProcess
from marquetry.transformers import Compose, Flatten, ToFloat, Normalize


# ===========================================================================
# Dataset base class
# ===========================================================================
class Dataset(object):
    def __init__(self, train=True, transform=None, target_transform=None, **kwargs):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.transform is None:
            self.transform = lambda x: x

        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.source = None
        self.target = None

        self._set_data(**kwargs)

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.target is None:
            return self.transform(self.source[index]), None
        else:
            return self.transform(self.source[index]), self.target_transform(self.target[index])

    def __len__(self):
        return len(self.source)

    def _set_data(self, *args, **kwargs):
        raise NotImplementedError()


# ===========================================================================
# MNIST / Fashion MNIST
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


class FashionMNIST(Dataset):
    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]), target_transform=None):
        super().__init__(train, transform, target_transform)

    def _set_data(self):
        url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

        train_files = {
            "source": "train-images-idx3-ubyte.gz",
            "target": "train-labels-idx1-ubyte.gz"
        }
        test_files = {
            "source": "t10k-images-idx3-ubyte.gz",
            "target": "t10k-labels-idx1-ubyte.gz"
        }

        files = train_files if self.train else test_files
        class_label = "train" if self.train else "test"
        source_path = get_file(url + files["source"], "fashion_{}_data.gz".format(class_label))
        target_path = get_file(url + files["target"], "fashion_{}_label.gz".format(class_label))

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
        return {
            0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
            4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}


# ===========================================================================
# Sequential dataset: SinCurve
# ===========================================================================
class SinCurve(Dataset):
    def _set_data(self):
        num_data = 5000
        dtype = np.float64

        x = np.linspace(0, 3 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)

        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)

        y = y.astype(dtype)
        self.target = y[1:][:, np.newaxis]
        self.source = y[:-1][:, np.newaxis]


# ===========================================================================
# Titanic
# ===========================================================================
class Titanic(Dataset):
    """
    Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.
    """
    def __init__(self, train=True, transform=ToFloat(), target_transform=None,
                 train_rate=0.8, auto_fillna=True, is_normalize=True, is_one_hot=True, **kwargs):
        self.train_rate = train_rate
        self.auto_fillna = auto_fillna
        self.is_one_hot = is_one_hot
        self.is_normalize = is_normalize

        self.target_columns = None
        self.source_columns = None
        self.drop_columns = kwargs.get("drop_columns", [])

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

        preprocess = LinearPreProcess(categorical_columns, numerical_columns)
        use_cache_params = self._valid_exists_params(preprocess, file_path)
        if not use_cache_params and not self.train:
            raise Exception("test data preprocess needs train statistic but the train statistic is not found.")

        titanic_df = preprocess(titanic_df, is_train=self.train, to_one_hot=self.is_one_hot,
                                to_normalize=self.is_normalize, fill_na=self.auto_fillna,
                                target_column="survived", model_mode="classification", **kwargs)
        index_series = pd.Series(titanic_df.index, name="index")
        titanic_df = pd.concat([index_series, titanic_df], axis=1)

        if not use_cache_params:
            self._save_params(preprocess, file_path)

        return titanic_df

    @staticmethod
    def _valid_exists_params(preprocess, file_path):
        param_file_name = file_path[file_path.rfind("/") + 1:]
        param_file_name = param_file_name.split(".")[0] + ".json"
        param_path = os.path.join(file_path[:file_path.rfind("/")], param_file_name)

        params = None
        if os.path.exists(param_path):
            with open(param_path, "r") as f:
                params = json.load(f)

        if params is not None:
            preprocess.load_params(params)
            return True
        else:
            return False

    @staticmethod
    def _save_params(preprocess, file_path):
        param_file_name = file_path[file_path.rfind("/") + 1:]
        param_file_name = param_file_name.split(".")[0] + ".json"
        param_path = os.path.join(file_path[:file_path.rfind("/")], param_file_name)

        params = preprocess.download_params()

        overload_handler = True
        if os.path.exists(param_path):
            ask_handler = True
            while ask_handler:
                is_overload = input(
                    "Params set {} is already existing, would you want to overload? (Y/n): ".format(param_file_name))

                if is_overload.lower() in ("y", "yes", "n", "no"):
                    if is_overload.lower() in ("y", "yes"):
                        overload_handler = True
                    else:
                        overload_handler = False

                    ask_handler = False

        if overload_handler:
            with open(param_path, "w") as f:
                json.dump(params, f)
        else:
            print("The param isn't overloaded due to the user input.")

    @property
    def categorical_columns(self):
        return ["pclass", "sex", "cabin", "embarked", "name", "ticket"]

    @property
    def numerical_columns(self):
        return ["age", "sibsp", "parch", "fare"]


# ===========================================================================
# CsvLoader: to use custom csv file
# ===========================================================================
class CsvLoader(Dataset):
    def __init__(self):
        super().__init__()

    def _set_data(self, *args, **kwargs):
        raise NotImplementedError()

import json
import os

import pandas as pd

from marquetry import configuration


# ===========================================================================
# preprocess base class
# ===========================================================================
class Preprocess(object):
    _label = None
    _msg = """if you use new data for the training, please use new `name` parameter or delete the old statistic data"""

    def __init__(self, name):
        self._name = name

        data_dir = os.path.join(configuration.config.CACHE_DIR, name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.data_dir = data_dir
        self._statistic_data = None

    def __call__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Preprocess requires the input is pandas DataFrame.")

        if self._statistic_data is not None:
            self._validate_structure(data)
        data.reset_index(drop=True, inplace=True)
        output = self.process(data)

        if isinstance(output, tuple):
            output = output[0]

        return output

    def process(self, data):
        raise NotImplementedError()

    def _validate_structure(self, data: pd.DataFrame):
        if not isinstance(self._statistic_data, dict):
            raise TypeError("statistic data is wrong, expected dict but got {}".format(type(self._statistic_data)))

        data_columns = set(data.columns)
        statistic_columns = set(self._statistic_data.keys())

        if data_columns != statistic_columns:
            raise ValueError("saved static data: {} is exist, but the construct is wrong".format(self._name))

        return

    def _save_statistic(self, statistic_data: dict):
        if self._label is None:
            raise NotImplementedError()

        file_name = self._name + "." + self._label + ".json"
        file_path = os.path.join(self.data_dir, file_name)

        with open(file_path, "w") as f:
            json.dump(statistic_data, f)

        return

    def _load_statistic(self):
        if self._label is None:
            raise NotImplementedError()

        file_name = self._name + "." + self._label + ".json"
        file_path = os.path.join(self.data_dir, file_name)

        if not os.path.exists(file_path):
            return None

        with open(file_path, "r") as f:
            statistic_data = json.load(f)

        return statistic_data

    def remove_old_statistic(self):
        file_name = self._name + "." + self._label + ".json"
        file_path = os.path.join(self.data_dir, file_name)

        if os.path.exists(file_path):
            os.remove(file_path)

        self._statistic_data = None

        return

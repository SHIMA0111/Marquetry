import json
import os
import warnings

import pandas as pd

from marquetry import configuration


# ===========================================================================
# preprocess base class
# ===========================================================================
class Preprocess(object):
    """Base class for preprocessing CSV or Datamart(Not implement now) data.

        All preprocess implementation defined in :mod:`marquetry.preprocesses` inherit
        this class.

        The main feature of this class is to provide a uniform process for all preprocessing steps.
        When a preprocess function receives input data, it first checks the data type and structure.
        After that, it resets the data index and performs the preprocessing.

        Attributes:
            data_dir (str): The directory where statistic data is stored.

        Args:
            name (str): A unique name for the preprocess instance.
                It is used for saving and loading statistic data.
            is_train(bool): Specifying the data is ``train`` or not.

    """

    _label = None
    _msg = """if you use new data for the training, please use new `name` parameter or delete the old statistic data"""

    def __init__(self, name, is_train):
        self._name = name
        self._is_train = is_train

        data_dir = os.path.join(configuration.config.CACHE_DIR, name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.data_dir = data_dir
        self._statistic_data, self._statistic_size = self._load_statistic()

    def __call__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Preprocess requires the input is pandas DataFrame.")

        if self._statistic_data is not None:
            self._validate_structure(data)

        if self._statistic_data is None and not self._is_train:
            raise ValueError("In test data process, train statistic must be needed but there is no statistic data.")

        if self._statistic_size is None:
            self._statistic_size = len(data.index)

        data.reset_index(drop=True, inplace=True)
        output = self.process(data)

        if isinstance(output, tuple):
            output = output[0]

        return output

    def process(self, data):
        """Define perform custom preprocessing on the input data. (to be implemented by subclasses)

            Args:
                data (pd.DataFrame): Input data in the form of a pandas DataFrame.

            Returns:
                pd.DataFrame: Preprocessed data.

        """

        raise NotImplementedError()

    def _validate_structure(self, data: pd.DataFrame):
        """Validate the structure of the input data compared to saved statistic data.

            Args:
                data (pd.DataFrame): Input data in the form of a pandas DataFrame.

        """
        if self._statistic_size != len(data) and self._is_train:
            self._statistic_data = None
            self.remove_old_statistic()
            self._statistic_size = len(data.index)

            return

        if not isinstance(self._statistic_data, dict):
            msg = "statistic data is wrong, expected dict but got {}. ".format(type(self._statistic_data))
            if not self._is_train:
                raise TypeError(msg)

            warnings.warn(msg + "Delete and Recalculate the statistic file.", SyntaxWarning)
            self._statistic_data = None
            self.remove_old_statistic()

            return

        data_columns = set(data.columns)
        statistic_columns = set(self._statistic_data.keys())

        if data_columns != statistic_columns:
            raise ValueError("Statistic data unmatch with input data. Please check correctness of the statistic name.")

        return

    def _save_statistic(self, statistic_data: dict):
        """Save statistic data to a file.

            Args:
                statistic_data (dict): Statistic data to be saved.

            Note:
                This method requires self._label is not None.
                Therefore, if this method call in the base class, always it failed by NotImplementedError.
        """

        if self._label is None:
            raise NotImplementedError()

        file_name = self._name + "." + self._label + ".json"
        file_path = os.path.join(self.data_dir, file_name)

        save_data_dict = {"statistic_data": statistic_data, "statistic_size": self._statistic_size}

        with open(file_path, "w") as f:
            json.dump(save_data_dict, f)

        return

    def _load_statistic(self):
        """Load saved statistic data from a file.

            Returns:
                dict or None: Loaded statistic data, or None if no data is found.

            Note:
                This method requires self._label is not None.
                Therefore, if this method call in the base class, always it failed by NotImplementedError.

        """

        if self._label is None:
            raise NotImplementedError()

        file_name = self._name + "." + self._label + ".json"
        file_path = os.path.join(self.data_dir, file_name)

        if not os.path.exists(file_path):
            return None, None

        with open(file_path, "r") as f:
            load_data = json.load(f)

        statistic_data = load_data["statistic_data"]
        data_size = load_data["statistic_size"]

        return statistic_data, data_size

    def remove_old_statistic(self):
        """Remove previously saved statistic data."""

        file_name = self._name + "." + self._label + ".json"
        file_path = os.path.join(self.data_dir, file_name)

        if os.path.exists(file_path):
            os.remove(file_path)

        self._statistic_data = None

        return

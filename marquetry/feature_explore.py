from typing import Optional

import numpy as np
import pandas as pd


class FeatureExplore(object):
    def __init__(self, model_type: str, categorical_columns: list, numerical_columns: list,
                 explore_limit=3000, select_limit=500):
        self.model_type = model_type
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.feature_nums = 0
        self.features = {}

    def categorical_feature_explore(self, data: pd.DataFrame):
        for categorical_column in self.categorical_columns:
            target_column = data.loc[:, categorical_column]

    def numerical_feature_explore(self, data: pd.DataFrame):
        pass

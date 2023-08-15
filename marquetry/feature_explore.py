import itertools
import sys
from typing import Optional

import numpy as np
import pandas as pd


class FeatureExplore(object):
    def __init__(self, model_type: str, categorical_columns: list, numerical_columns: list, target_column: str,
                 explore_num_limit=500, select_limit=500):
        if model_type.lower() not in ("classification", "regression"):
            raise TypeError("You can use only `classification` or `regression` but you input {}.".format(model_type))

        self.model_type = model_type.lower()
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.target_column = target_column

        self.select_limit = select_limit
        self.explore_num_limit = explore_num_limit

        self.feature_nums = 0
        self.features = {}
        self.feature_explanation = {}
        self.generated_explanation = {}
        self.generated_eval = {}
        self.original_statistic = {}
        self.feature_gen_only = False

        self.base_target_eval = None
        self.data_uniques = None

    def __call__(self, x: pd.DataFrame):
        if len(self.feature_explanation) == 0:
            self.base_target_eval = x[self.target_column].values.sum() / len(x)
            self.data_uniques = x.nunique()
            self.categorical_feature_explore(x)
            self.numerical_feature_explore(x)
            self.feature_select()
        self.generate_features(x)
        feature_table = self.avoid_multiply_collinearity()
        if isinstance(feature_table, pd.Series):
            index_series = pd.Series(list(range(len(feature_table))))
        else:
            index_series = pd.Series(feature_table.index, name="index")
        feature_table = pd.concat((index_series, feature_table, x[self.target_column]), axis=1)

        return feature_table

    def categorical_feature_explore(self, data: pd.DataFrame):
        for categorical_column in self.categorical_columns:
            explore_column = data.loc[:, (categorical_column, self.target_column)]
            unique_set = tuple(set(explore_column[categorical_column].astype(int)))

            for unique_value in unique_set:
                explanation = "categorical,{},{}".format(categorical_column, unique_value)
                feature_name = "feature_" + str(self.feature_nums)

                unique_feature = explore_column[self.target_column].loc[
                    explore_column[categorical_column] == unique_value]

                if len(unique_feature) < 10:
                    continue

                eval_value = unique_feature.values.sum() / len(unique_feature)
                self.generated_explanation[feature_name] = explanation
                self.generated_eval[feature_name] = eval_value

                self.feature_nums += 1

    def numerical_feature_explore(self, data: pd.DataFrame):
        data_size = len(data)

        for numerical_column in self.numerical_columns:
            explore_column: pd.DataFrame = data.loc[:, (numerical_column, self.target_column)]

            column_std = explore_column[numerical_column].values.std()
            column_mean = explore_column[numerical_column].values.mean()

            self.original_statistic[numerical_column] = {"std": column_std, "mean": column_mean}

            sd_max, sd_min = column_mean + 2 * column_std, column_mean - 2 * column_std
            max_value, min_value = explore_column[numerical_column].max(), explore_column[numerical_column].min()

            min_value = max(min_value, sd_min)
            max_value = min(max_value, sd_max)

            feature_explore_range = np.linspace(min_value, max_value, num=self.explore_num_limit)

            pre_eval = 0
            for i in range(len(feature_explore_range) - 1):
                min_value = feature_explore_range[i]

                min_query_text = str(min_value) + " <= " + numerical_column

                min_explanation = "numerical,{},{}".format(numerical_column, min_query_text)
                min_feature_name = "feature_" + str(self.feature_nums)

                min_range_feature = explore_column.query(min_query_text)

                if len(min_range_feature) < data_size * 0.01:
                    continue

                target_eval = min_range_feature[self.target_column].values.sum() / len(min_range_feature)

                if target_eval == pre_eval:
                    continue

                pre_eval = target_eval

                self.generated_explanation[min_feature_name] = min_explanation
                self.generated_eval[min_feature_name] = target_eval

                self.feature_nums += 1

            pre_eval = 0
            for i in range(len(feature_explore_range) - 1):
                max_value = feature_explore_range[::-1][i]

                max_query_text = str(max_value) + " >= " + numerical_column

                max_explanation = "numerical,{},{}".format(numerical_column, max_query_text)
                max_feature_name = "feature_" + str(self.feature_nums)

                max_range_feature = explore_column.query(max_query_text)

                if len(max_range_feature) < data_size * 0.01:
                    continue

                target_eval = max_range_feature[self.target_column].values.sum() / len(max_range_feature)

                if target_eval == pre_eval:
                    continue

                pre_eval = target_eval

                self.generated_explanation[max_feature_name] = max_explanation
                self.generated_eval[max_feature_name] = target_eval

                self.feature_nums += 1

    def feature_select(self):
        feature_evals = pd.Series(self.generated_eval)
        diff_feature_evals = (self.base_target_eval - feature_evals).abs()
        features = diff_feature_evals.loc[diff_feature_evals > 0.10]
        selected_feature_explanation = {key: self.generated_explanation[key] for key in features.keys()}

        for categorical_name in self.categorical_columns:
            unique_num = self.data_uniques[categorical_name]
            tmp_explanation = {
                key: explanation for key, explanation in selected_feature_explanation.items()
                if categorical_name in explanation
            }
            if len(tmp_explanation) == 0:
                continue

            if unique_num == len(tmp_explanation):
                tmp_explanation.popitem()

            self.feature_explanation.update(tmp_explanation)

        for numerical_name in self.numerical_columns:
            tmp_explanation = {
                key: explanation for key, explanation in selected_feature_explanation.items()
                if numerical_name in explanation
            }

            if len(tmp_explanation) == 0:
                continue

            tmp_eval_value = {
                key: diff_feature_evals[key] for key in tmp_explanation.keys()
            }

            tmp_eval_value = dict(sorted(tmp_eval_value.items(), key=lambda x: x[1]))

            pre_lt_range, pre_lt_eval = float("-inf"), float("-inf")
            pre_gt_range, pre_gt_eval = float("inf"), float("-inf")
            column_std, column_mean = self.original_statistic[numerical_name].values()

            for feature_name, eval_value in tmp_eval_value.items():
                condition = tmp_explanation[feature_name].split(",")[2]

                if ">=" in condition:
                    range_value = float(condition.split(" ")[0])

                    if range_value >= pre_lt_range and eval_value >= pre_lt_eval:
                        pre_lt_range, pre_lt_eval = range_value, eval_value
                    elif eval_value > pre_lt_eval:
                        if (column_mean - column_std) < range_value:
                            pre_lt_range, pre_lt_eval = range_value, eval_value
                        elif ((column_mean - 2 * column_std) < range_value and
                              not (column_mean - column_std) < pre_lt_range):
                            pre_lt_range, pre_lt_eval = range_value, eval_value
                        else:
                            pass
                    elif range_value > pre_lt_range:
                        diff_evals = pre_lt_eval - eval_value
                        diff_range = range_value - pre_lt_range

                        if diff_evals < 0.01 or (diff_evals < 0.05 and (column_std / 4) < diff_range):
                            pre_gt_range, pre_gt_eval = range_value, eval_value
                        else:
                            pass

                elif "<=" in condition:
                    range_value = float(condition.split(" ")[0])

                    if range_value <= pre_gt_range and eval_value >= pre_gt_eval:
                        pre_gt_range, pre_gt_eval = range_value, eval_value
                    elif eval_value > pre_gt_eval:
                        if range_value < (column_mean + column_std):
                            pre_gt_range, pre_gt_eval = range_value, eval_value
                        elif (range_value < (column_mean + 2 * column_std) and
                              not pre_gt_range < (column_mean + column_std)):
                            pre_gt_range, pre_gt_eval = range_value, eval_value
                        else:
                            pass
                    elif range_value < pre_gt_range:
                        diff_evals = pre_gt_eval - eval_value
                        diff_range = pre_gt_range - range_value

                        if diff_evals < 0.01 or (diff_evals < 0.05 and (column_std / 4) < diff_range):
                            pre_gt_range, pre_gt_eval = range_value, eval_value
                        else:
                            pass

            tmp_dict = {}
            lt_feature = {key: tmp_explanation[key] for key, value in tmp_eval_value.items()
                          if value == pre_lt_eval and float(tmp_explanation[key].split(",")[2].split(" ")[0]) == pre_lt_range}
            gt_feature = {key: tmp_explanation[key] for key, value in tmp_eval_value.items()
                          if value == pre_gt_eval and float(tmp_explanation[key].split(",")[2].split(" ")[0]) == pre_gt_range}
            tmp_dict.update(lt_feature)
            tmp_dict.update(gt_feature)

            self.feature_explanation.update(tmp_dict)

    def generate_features(self, data: pd.DataFrame):
        for feature_name, explanation in self.feature_explanation.items():
            if explanation == "original_feature":
                self.features[feature_name] = len(data[feature_name].values)
                continue
            feature_type, column_name, condition = explanation.split(",")

            if feature_type == "categorical":
                new_feature = (data[column_name].astype(int) == int(condition))

            elif feature_type == "numerical":
                condition_value = float(condition.split(" ")[0])
                if ">=" in condition:
                    new_feature = (data[column_name].astype(float) <= condition_value)
                else:
                    new_feature = (data[column_name].astype(float) >= condition_value)

            else:
                raise Exception("Unexpected exception occurred when")

            new_feature = new_feature.astype(int)
            self.features[feature_name] = list(new_feature.values)

        for numerical_name in self.numerical_columns:
            tmp_calc_column = data.loc[:, (numerical_name, self.target_column)]
            corr_data = tmp_calc_column.corr().iloc[0, 1]
            self.generated_eval[numerical_name] = corr_data
            if abs(corr_data) >= 0.2 and (not self.feature_gen_only):
                self.feature_explanation[numerical_name] = "original_feature"
                self.features[numerical_name] = list(tmp_calc_column[numerical_name].values)

    def avoid_multiply_collinearity(self):
        if len(self.features) == 1:
            return pd.DataFrame(self.features)
        tmp_df = pd.DataFrame(self.features)
        corr_data = tmp_df.corr()

        header_list = list(tmp_df.keys())
        header_index = itertools.combinations(header_list, 2)

        drop_set = set()
        for idx in header_index:
            corr_value = corr_data.loc[idx[0], idx[1]]
            if abs(corr_value) > 0.8:
                if idx[1] not in drop_set:
                    drop_set.add(idx[1])

        drop_set = list(drop_set)
        tmp_df = tmp_df.drop(drop_set, axis=1)

        self.generated_eval = {key: value for key, value in self.generated_eval.items() if key not in drop_set}
        self.feature_explanation = {key: value for key, value in self.feature_explanation.items() if key not in drop_set}

        return tmp_df

    def download_feature(self):
        return {
            "feature_explanation": self.feature_explanation,

            "all_features": {
                "generated_features": self.generated_explanation,
                "generated_evals": self.generated_eval
            }
        }

    def load_feature(self, feature_explanation: dict, use_unselected: list = None):
        try:
            self.feature_explanation = feature_explanation["feature_explanation"]
            self.feature_gen_only = True
            if use_unselected is not None:
                if not isinstance(use_unselected, (list, tuple)):
                    use_unselected = (use_unselected,)
                unuse_feature_dict = feature_explanation["all_features"]["generated_features"]

                for feature_name in use_unselected:
                    selected_feature_explanation = unuse_feature_dict.get(feature_name)
                    if selected_feature_explanation is None:
                        print("!!!Skipped!!! {} is not found the feature pool.".format(feature_name))

                    add_tmp_dict = {feature_name: selected_feature_explanation}

                    self.feature_explanation.update(add_tmp_dict)

        except Exception as e:
            raise Exception("Your feature file seems to be broken, please check and fix it.")

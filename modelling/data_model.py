import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:

        y = df.y.to_numpy()
        y_series = pd.Series(y)

        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index

        if len(good_y_value) < 1:
            print("None of the class have more than 3 records: Skipping ...")
            self.X_train = None
            return

        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]

        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_good, y_good, test_size=new_test_size, random_state=0, stratify=y_good
        )
        self.y = y_good
        self.classes = good_y_value
        self.embeddings = X

    def get_type(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_train_df(self):
        return self.train_df

    def get_embeddings(self):
        return self.embeddings

    def get_type_test_df(self):
        return self.test_df

    def get_X_DL_test(self):
        return self.X_DL_test

    def get_X_DL_train(self):
        return self.X_DL_train

"""
Author: Sharath Cherry

ChainedData is the data encapsulation class designed for Design Choice 1:
Chained Multi-Output Classification. It reads the y2, y3, and y4 label
columns from the input DataFrame and builds three progressively combined
target variables. A single stratified train/test split is then performed
on the Type 2 label (y2) so that all three chain levels share the same
split, ensuring evaluation is consistent and comparable across levels.
""" 

class ChainedData:
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        df = df.copy().reset_index(drop=True)

        for col in ['y2', 'y3', 'y4']:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.strip()
            else:
                df[col] = ''

        df['y_l1'] = df['y2']
        df['y_l2'] = df['y2'] + ' | ' + df['y3']
        df['y_l3'] = df['y2'] + ' | ' + df['y3'] + ' | ' + df['y4']

        y2 = df['y2']
        counts = y2.value_counts()
        good_y2 = counts[counts >= 3].index

        if len(good_y2) < 1:
            print("Not enough Type 2 classes (need >= 3 instances each). Skipping group...")
            self.X_train = None
            return

        mask = y2.isin(good_y2) & (y2 != '')
        mask_arr = mask.values

        y2_good = y2[mask].values
        X_good = X[mask_arr]
        df_good = df[mask].reset_index(drop=True)

        if X_good.shape[0] == 0:
            print("No valid samples after filtering. Skipping group...")
            self.X_train = None
            return

        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]
        new_test_size = min(max(new_test_size, 0.1), 0.4)

        try:
            train_idx, test_idx = train_test_split(
                np.arange(len(df_good)),
                test_size=new_test_size,
                random_state=0,
                stratify=y2_good
            )
        except ValueError as e:
            print(f"Stratified split failed ({e}). Using random split.")
            train_idx, test_idx = train_test_split(
                np.arange(len(df_good)),
                test_size=new_test_size,
                random_state=0
            )

        self.X_train = X_good[train_idx]
        self.X_test = X_good[test_idx]
        self.embeddings = X

        for lvl in ['l1', 'l2', 'l3']:
            col = f'y_{lvl}'
            arr = df_good[col].values
            setattr(self, f'y_train_{lvl}', arr[train_idx])
            setattr(self, f'y_test_{lvl}', arr[test_idx])

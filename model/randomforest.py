import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random
num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions

    def print_results(self, data):
        print(classification_report(data.y_test, self.predictions))

    def data_transform(self) -> None:
        ...


'''
Author: Rizvi Abbas
ChainedRandomForest implements Design Choice 1: Chained Multi-Output
Classification. Three separate RandomForestClassifier instances are
maintained internally, each trained on a progressively combined target:
   Level 1 — Type 2 alone
   Level 2 — Type 2 combined with Type 3
   Level 3 — Type 2 combined with Type 3 and Type 4

All three classifiers share the same TF-IDF feature matrix, ensuring
the only variable across models is the target label. The train(),
predict(), and print_results() methods follow the same interface defined
in BaseModel, so the controller interacts with this class identically
to any other model implementation in the system.
'''

class ChainedRandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(ChainedRandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y

        rf_params = dict(n_estimators=500, random_state=seed, class_weight='balanced_subsample')
        self.mdl_l1 = RandomForestClassifier(**rf_params)
        self.mdl_l2 = RandomForestClassifier(**rf_params)
        self.mdl_l3 = RandomForestClassifier(**rf_params)

        self.predictions_l1 = None
        self.predictions_l2 = None
        self.predictions_l3 = None

        self.data_transform()

    def train(self, data) -> None:
        print("  Training Level 1 classifier (Type 2 only)...")
        self.mdl_l1.fit(data.X_train, data.y_train_l1)

        print("  Training Level 2 classifier (Type 2 + Type 3)...")
        self.mdl_l2.fit(data.X_train, data.y_train_l2)

        print("  Training Level 3 classifier (Type 2 + Type 3 + Type 4)...")
        self.mdl_l3.fit(data.X_train, data.y_train_l3)

    def predict(self, X_test: np.ndarray) -> None:
        self.predictions_l1 = self.mdl_l1.predict(X_test)
        self.predictions_l2 = self.mdl_l2.predict(X_test)
        self.predictions_l3 = self.mdl_l3.predict(X_test)

    def print_results(self, data) -> None:
        separator = "=" * 70

        print(f"\n{separator}")
        print("Chain Level 1 — Classifying Type 2 only")
        print(separator)
        print(classification_report(data.y_test_l1, self.predictions_l1, zero_division=0))

        print(f"\n{separator}")
        print("Chain Level 2 — Classifying Type 2 + Type 3 (combined label)")
        print(separator)
        print(classification_report(data.y_test_l2, self.predictions_l2, zero_division=0))

        print(f"\n{separator}")
        print("Chain Level 3 — Classifying Type 2 + Type 3 + Type 4 (combined label)")
        print(separator)
        print(classification_report(data.y_test_l3, self.predictions_l3, zero_division=0))

    def data_transform(self) -> None:
        ...

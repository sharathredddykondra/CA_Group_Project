from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    # load the input data
    df = get_input_data()
    return df

def preprocess_data(df):
    # De-duplicate input data
    df = de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)


# =============================================================================
# Author: Sharath Cherry
#
# The following functions extend the existing pipeline to support Design
# Choice 1: Chained Multi-Output Classification. get_chained_data_object()
# wraps ChainedData construction, and perform_chained_modelling() delegates
# to model_predict_chained() in modelling.py, keeping the main controller
# free of any modelling or data-encapsulation logic.
# =============================================================================
def get_chained_data_object(X: np.ndarray, df: pd.DataFrame):
    return ChainedData(X, df)

def perform_chained_modelling(data: ChainedData, df: pd.DataFrame, name):
    model_predict_chained(data, df, name)


if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)

    # -------------------------------------------------------------------------
    # Original pipeline — single-label classification on Type 2 (Exercise 3)
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Original Pipeline — Single-Label Classification (Type 2)")
    print("=" * 70)
    for name, group_df in grouped_df:
        print(name)
        X, group_df = get_embeddings(group_df)
        data = get_data_object(X, group_df)
        perform_modelling(data, group_df, name)

    # -------------------------------------------------------------------------
    # Design Choice 1 — Chained Multi-Output Classification
    # Author: Sharath Cherry
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Design Choice 1 — Chained Multi-Output Classification")
    print("=" * 70)
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print(f"\nGroup: {name}")
        X, group_df = get_embeddings(group_df)
        chained_data = get_chained_data_object(X, group_df)
        if chained_data.X_train is None:
            print(f"Skipping group '{name}' — insufficient data.")
            continue
        perform_chained_modelling(chained_data, group_df, name)

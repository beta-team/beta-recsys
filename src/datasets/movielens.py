"""
Created on 2020.02.17 BY @zaiqiao

------

@zaiqiao: Zaiqiao Meng (zaiqiao.meng@gmail.com)

"""
import sys
import os
import math
import random
import numpy as np
import pandas as pd
import sklearn

sys.path.append("../")

import utils.constants as Constants
from utils.unigramTable import UnigramTable

# indicators of the colunmn name
DEFAULT_USER_COL = Constants.DEFAULT_USER_COL
DEFAULT_ITEM_COL = Constants.DEFAULT_ITEM_COL
DEFAULT_ORDER_COL = Constants.DEFAULT_ORDER_COL
DEFAULT_RATING_COL = Constants.DEFAULT_RATING_COL
DEFAULT_LABEL_COL = Constants.DEFAULT_LABEL_COL
DEFAULT_TIMESTAMP_COL = Constants.DEFAULT_TIMESTAMP_COL
DEFAULT_PREDICTION_COL = Constants.DEFAULT_PREDICTION_COL
DEFAULT_FLAG_COL = Constants.DEFAULT_FLAG_COL

par_abs_dir = os.path.abspath(os.path.join(os.path.abspath("."), os.pardir))

# raw dataset
ml_1m_raw_dir = "datasets/ml-1m/raw/ratings.dat"
# dataset dir under temporal split
ml_1m_temporal_dir = "datasets/ml-1m/temporal"
# dataset dir under leave-one-out split
ml_1m_l1o_dir = os.path.join(par_abs_dir, "datasets/ml-1m/leave_one_out")


def load_data(data_dir, max_id=0):
    loaded = np.load(os.path.join(data_dir, "train.npz"))
    train_df = pd.DataFrame(
        data={
            DEFAULT_USER_COL: loaded["user_ids"],
            DEFAULT_ITEM_COL: loaded["item_ids"],
            DEFAULT_RATING_COL: loaded["ratings"],
            DEFAULT_TIMESTAMP_COL: loaded["timestamp"],
        }
    )
    if max_id:
        train_df = train_df[
            (train_df[DEFAULT_USER_COL] < max_id)
            & (train_df[DEFAULT_ITEM_COL] < max_id)
        ]
    valid_dfs = []
    test_dfs = []
    for i in range(10):
        loaded = np.load(os.path.join(data_dir, "valid_" + str(i) + ".npz"))
        valid_df = pd.DataFrame(
            data={
                DEFAULT_USER_COL: loaded["user_ids"],
                DEFAULT_ITEM_COL: loaded["item_ids"],
                DEFAULT_RATING_COL: loaded["ratings"],
            }
        )
        if max_id:
            valid_df = valid_df[
                (valid_df[DEFAULT_USER_COL] < max_id)
                & (valid_df[DEFAULT_ITEM_COL] < max_id)
            ]
        loaded = np.load(os.path.join(data_dir, "test_" + str(i) + ".npz"))
        test_df = pd.DataFrame(
            data={
                DEFAULT_USER_COL: loaded["user_ids"],
                DEFAULT_ITEM_COL: loaded["item_ids"],
                DEFAULT_RATING_COL: loaded["ratings"],
            }
        )
        if max_id:
            test_df = test_df[
                (test_df[DEFAULT_USER_COL] < max_id)
                & (test_df[DEFAULT_ITEM_COL] < max_id)
            ]
        valid_dfs.append(valid_df)
        test_dfs.append(test_df)
    return train_df, valid_dfs, test_dfs


def load_raw(root_dir=par_abs_dir):
    data_file = os.path.join(par_abs_dir, ml_1m_raw_dir)
    print("loading ml-1m raw dataset")
    ml1m_rating = pd.read_csv(
        data_file,
        sep="::",
        header=None,
        names=["uid", "mid", "rating", "timestamp"],
        engine="python",
    )
    data_df = ml1m_rating.rename(
        columns={
            "uid": DEFAULT_USER_COL,
            "mid": DEFAULT_ITEM_COL,
            "rating": DEFAULT_RATING_COL,
            "timestamp": DEFAULT_TIMESTAMP_COL,
        }
    )
    return data_df


def load_leave_one_out(root_dir=par_abs_dir, max_id=0):
    data_file = os.path.join(root_dir, ml_1m_l1o_dir)
    print("loading ml-1m dataset using leave_one_out split")
    return load_data(data_file, max_id)


def load_temporal(root_dir=par_abs_dir, max_id=0):
    data_file = os.path.join(root_dir, ml_1m_temporal_dir)
    print("loading ml-1m dataset using temporal split")
    return load_data(data_file, max_id)
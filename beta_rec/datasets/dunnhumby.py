import os
import numpy as np
import pandas as pd
from beta_rec.utils.constants import *

par_abs_dir = os.path.abspath(os.path.join(os.path.abspath("."), os.pardir))


"""
load dunnhumby raw data from a base dataset dirctory
"""


def load_raw(data_base_dir):

    transaction_data = (
        data_base_dir
        + "raw/dunnhumby_The-Complete-Journey/csv/"
        + "transaction_data.csv"
    )

    prior_transaction = pd.read_csv(
        transaction_data,
        usecols=["BASKET_ID", "household_key", "PRODUCT_ID", "DAY", "TRANS_TIME"],
    )

    prior_transaction["DAY"] = prior_transaction["DAY"].astype(str)  #
    prior_transaction["TRANS_TIME"] = prior_transaction["TRANS_TIME"].astype(str)

    prior_transaction["time"] = (
        prior_transaction["DAY"] + prior_transaction["TRANS_TIME"]
    )
    prior_transaction["time"] = prior_transaction["time"].astype(int)  #
    prior_transaction.reset_index(inplace=True)
    prior_transaction = prior_transaction.sort_values(by="time", ascending=False)

    prior_transaction.drop(["DAY", "TRANS_TIME"], axis=1)

    prior_transaction = prior_transaction[
        ["BASKET_ID", "household_key", "PRODUCT_ID", "time"]
    ]
    prior_transaction.insert(3, "flag", "train")
    prior_transaction.insert(4, "ratings", 1)
    prior_transaction.rename(
        columns={
            "BASKET_ID": DEFAULT_ORDER_COL,
            "household_key": DEFAULT_USER_COL,
            "PRODUCT_ID": DEFAULT_ITEM_COL,
            "flag": DEFAULT_FLAG_COL,
            "ratings": DEFAULT_RATING_COL,
            "time": DEFAULT_TIMESTAMP_COL,
        },
        inplace=True,
    )

    print("loading raw data completed")
    return prior_transaction


def load_data(data_dir, max_id=0, leave_one_item=False):
    loaded = np.load(os.path.join(data_dir, "train.npz"))
    train_df = pd.DataFrame(
        data={
            DEFAULT_USER_COL: loaded["user_ids"],
            DEFAULT_ITEM_COL: loaded["item_ids"],
            DEFAULT_ORDER_COL: loaded["order_ids"],
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
        if leave_one_item:
            valid_df = valid_df[valid_df[DEFAULT_RATING_COL] != 1].append(
                valid_df[valid_df[DEFAULT_RATING_COL] == 1].drop_duplicates(
                    [DEFAULT_USER_COL, DEFAULT_RATING_COL]
                )
            )
            test_df = test_df[test_df[DEFAULT_RATING_COL] != 1].append(
                test_df[test_df[DEFAULT_RATING_COL] == 1].drop_duplicates(
                    [DEFAULT_USER_COL, DEFAULT_RATING_COL]
                )
            )
        valid_dfs.append(valid_df)
        test_dfs.append(test_df)
    return train_df, valid_dfs, test_dfs


def load_item_fea_dic(data_base_dir, fea_type="w2c"):
    """
    Load item_feature_one.csv item_feature_w2v.csv item_feature_bert.csv

    Returns
    -------
    item_feature dict.

    """
    print("load item featrue for dataset:", data_str, " type:", fea_type)

    if fea_type == "word2vec":
        item_feature_file = open(data_base_dir + "/raw/item_feature_w2v.csv", "r")
    elif fea_type == "cate":
        item_feature_file = open(data_base_dir + "/raw/item_feature_cate.csv", "r")
    elif fea_type == "one_hot":
        item_feature_file = open(data_base_dir + "/raw/item_feature_one.csv", "r")
    elif fea_type == "bert":
        item_feature_file = open(data_base_dir + "/raw/item_feature_bert.csv", "r")
    else:
        print(
            "[ERROR]: CANNOT support other feature type, use 'random' user featrue instead!"
        )

    item_feature = {}
    lines = item_feature_file.readlines()
    for index in range(1, len(lines)):
        key_value = lines[index].split(",")
        item_id = int(key_value[0])
        feature = np.array(key_value[1].split(" "), dtype=np.float)
        item_feature[item_id] = feature
    return item_feature


def load_leave_one_item(root_dir=par_abs_dir, max_id=0):
    leave_one_out_dir = "datasets/dunnhumby/leave_one_item"
    data_file = os.path.join(root_dir, leave_one_out_dir)
    if not os.path.exists(data_file):
        raise RuntimeError(
            f"please download the dataset by your self and put it into {data_file}"
        )
    print("loading dunnhumby dataset using leave_one_item split")
    return load_data(data_file, max_id, leave_one_item=True)


def load_leave_one_out(root_dir=par_abs_dir, max_id=0):
    leave_one_out_dir = "datasets/dunnhumby/leave_one_basket"
    data_file = os.path.join(root_dir, leave_one_out_dir)
    if not os.path.exists(data_file):
        raise RuntimeError(
            f"please download the dataset by your self and put it into {data_file}"
        )
    print("loading dunnhumby dataset using leave_one_out split")
    return load_data(data_file, max_id)


def load_leave_one_basket(root_dir=par_abs_dir, max_id=0):
    leave_one_out_dir = "datasets/dunnhumby/leave_one_basket"
    data_file = os.path.join(root_dir, leave_one_out_dir)
    if not os.path.exists(data_file):
        raise RuntimeError(
            f"please download the dataset by your self and put it into {data_file}"
        )
    print("loading dunnhumby dataset using leave_one_basket split")
    return load_data(data_file, max_id)


def load_temporal(root_dir=par_abs_dir, max_id=0):
    temporal_dir = "datasets/dunnhumby/temporal"
    data_file = os.path.join(root_dir, temporal_dir)
    if not os.path.exists(data_file):
        raise RuntimeError(
            f"please download the dataset by your self and put it into {data_file}"
        )
    print("loading dunnhumby dataset using temporal split")
    return load_data(data_file, max_id)

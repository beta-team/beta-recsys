import os
import numpy as np
import pandas as pd

from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase, TAFENG_URL

par_abs_dir = os.path.abspath(os.path.join(os.path.abspath("."), os.pardir))


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


def load_leave_one_item(root_dir=par_abs_dir, max_id=0):
    leave_one_out_dir = "datasets/tafeng/leave_one_item"
    data_file = os.path.join(root_dir, leave_one_out_dir)
    print("loading ta-feng dataset using leave_one_item split")
    return load_data(data_file, max_id, leave_one_item=True)


def load_leave_one_out(root_dir=par_abs_dir, max_id=0):
    leave_one_out_dir = "datasets/tafeng/leave_one_basket"
    data_file = os.path.join(root_dir, leave_one_out_dir)
    print("loading ta-feng dataset using leave_one_out split")
    return load_data(data_file, max_id)


def load_leave_one_basket(root_dir=par_abs_dir, max_id=0):
    leave_one_out_dir = "datasets/tafeng/leave_one_basket"
    data_file = os.path.join(root_dir, leave_one_out_dir)
    print("loading ta-feng dataset using leave_one_basket split")
    return load_data(data_file, max_id)


def load_temporal(root_dir=par_abs_dir, max_id=0, test_percent=None):
    temporal_dir = "datasets/tafeng/temporal"
    print("loading ta-feng dataset using temporal split")
    data_file = os.path.join(root_dir, temporal_dir)
    if test_percent is not None:
        if test_percent < 1:
            test_percent = int(test_percent * 100)
        data_file = os.path.join(data_file, str(test_percent))
        print("split percent:", test_percent)
    return load_data(data_file, max_id)


class Tafeng(DatasetBase):
    def __init__(self):
        """Tafeng

        Tafeng dataset.
        The dataset can not be download by the url,
        you need to down the dataset by 'https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset/download'
        then put it into the directory `tafeng/raw`
        """
        super().__init__('tafeng',\
            manual_download_url='https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset/download'
        )

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = os.path.join(self.raw_path, 'ta_feng_all_months_merged.csv')
        if not os.path.exists(file_name):
            self.download()
        data = pd.read_table(
            file_name,
            sep=',',
            usecols=[0, 1, 5, 6],
            names=[DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL],
            header=0
        )

        data[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(data[DEFAULT_TIMESTAMP_COL])
        data[DEFAULT_TIMESTAMP_COL] = data[DEFAULT_TIMESTAMP_COL].map(lambda x: x.timestamp())

        self.save_dataframe_as_npz(data, os.path.join(self.processed_path, f'{self.dataset_name}_interaction.npz'))

import os
import numpy as np
import pandas as pd

from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase

# download_url
ML_100K_URL = r'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
ML_1M_URL = r'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
ML_25M_URL = r'http://files.grouplens.org/datasets/movielens/ml-25m.zip'

# indicators of the colunmn name
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


class Movielens_100k(DatasetBase):
    def __init__(self):
        """Movielens 100k

        Movielens 100k dataset.
        """
        super().__init__('ml_100k', url=ML_100K_URL)
    
    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, self.dataset_name, 'u.data')
        if not os.path.exists(file_name):
            self.download()

        data = pd.read_table(
            file_name,
            header=None,
            sep='\s+',
            engine='python',
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL]
            
        )
        self.save_dataframe_as_npz(data, os.path.join(self.processed_path, f'{self.dataset_name}_interaction.npz'))


class Movielens_1m(DatasetBase):
    def __init__(self):
        """Movielens 1m

        Movielens 1m dataset.
        """
        super().__init__('ml_1m', url=ML_1M_URL)
    
    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, self.dataset_name, 'ratings.dat')
        if not os.path.exists(file_name):
            self.download()

        data = pd.read_table(
            file_name,
            header=None,
            sep='::',
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL]
        )
        self.save_dataframe_as_npz(data, os.path.join(self.processed_path, f'{self.dataset_name}_interaction.npz'))


class Movielens_25m(DatasetBase):
    def __init__(self):
        """Movielens 25m

        Movielens 25m dataset.
        """
        super().__init__('ml_25m', url=ML_25M_URL)
    
    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, self.dataset_name, 'ratings.csv')
        if not os.path.exists(file_name):
            self.download()

        data = pd.read_table(
            file_name,
            header=None,
            sep=',',
            skiprows=[0],
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL]
        )
        self.save_dataframe_as_npz(data, os.path.join(self.processed_path, f'{self.dataset_name}_interaction.npz'))

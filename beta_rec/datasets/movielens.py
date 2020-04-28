import os
import numpy as np
import pandas as pd

from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase

# download_url
ML_100K_URL = r"http://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_1M_URL = r"http://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML_25M_URL = r"http://files.grouplens.org/datasets/movielens/ml-25m.zip"

# processed data url
ML_100K_LEAVE_ONE_OUT_URL = r"https://1drv.ms/u/s!AjMahLyQeZqugU-siALoN5y9eaCq?e=jsgoOB"
ML_100K_RANDOM_URL = r"https://1drv.ms/u/s!AjMahLyQeZqugVD4bv1iR6KgZn63?e=89eToa"
ML_100K_TEMPORAL_URL = r"https://1drv.ms/u/s!AjMahLyQeZqugVG_vS_DggoFaySY?e=HpcD9b"

ML_1M_LEAVE_ONE_OUT_URL = r"https://1drv.ms/u/s!AjMahLyQeZqugVMZ5TK2sTGBUSr0?e=32CmFJ"
ML_1M_RANDOM_URL = r"https://1drv.ms/u/s!AjMahLyQeZqugVW2Bl1A1kORNuTY?e=iEabat"
ML_1M_TEMPORAL_URL = r"https://1drv.ms/u/s!AjMahLyQeZqugVf8PRlo82hSnblP?e=VpZa0L"

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
        super().__init__(
            "ml_100k",
            url=ML_100K_URL,
            processed_leave_one_out_url=ML_100K_LEAVE_ONE_OUT_URL,
            processed_random_split_url=ML_100K_RANDOM_URL,
            processed_temporal_split_url=ML_100K_TEMPORAL_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, self.dataset_name, "u.data")
        if not os.path.exists(file_name):
            self.download()

        data = pd.read_table(
            file_name,
            header=None,
            sep="\s+",
            engine="python",
            names=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
        )
        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

    def make_fea_vec(self):
        """Make feature vectors for users and items.
        1. For items (movies), we use the last 19 fields as feature, which are the genres, 
        with 1 indicateing the movie is of that genre, and 0 indicateing it is not; 
        movies can be in several genres at once.
        
        2. For users, we construct one_hot encoding for age, gender and occupation as their
        feature, where ages are categorized into 8 groups.
        
        Returns:
            user_feat (numpy.ndarray): The first column is the user id, rest column are feat vectors
            item_feat (numpy.ndarray): The first column is the item id, rest column are feat vectors
        
        """
        print(f"Making user and item feature vactors for dataset {self.dataset_name}")
        data = pd.read_table(
            f"{self.dataset_dir}/raw/ml_100k/u.item",
            header=None,
            sep="|",
            engine="python",
        )
        item_feat = data[[0] + [i for i in range(5, 24)]].to_numpy()
        # first column is the item id, other 19 columns are feature
        data = pd.read_table(
            f"{self.dataset_dir}/raw/ml_100k/u.user",
            header=None,
            sep="|",
            engine="python",
        )
        age_one_hot = np.eye(8).astype(np.int)
        # categorize age into 8 groups
        age_maping = {
            1: age_one_hot[0],
            2: age_one_hot[1],
            3: age_one_hot[2],
            4: age_one_hot[3],
            5: age_one_hot[4],
            6: age_one_hot[5],
            7: age_one_hot[6],
            8: age_one_hot[7],
        }
        data["age_one_hot"] = data[1].apply(lambda x: age_maping[int(x / 10) + 1])
        col_2 = data[2].unique()
        col_2_one_hot = np.eye(len(col_2)).astype(np.int)
        col_2_maping = {}
        for idx, col in enumerate(col_2):
            col_2_maping[col] = col_2_one_hot[idx]
        data["col_2_one_hot"] = data[2].apply(lambda x: col_2_maping[x])
        col_3 = data[3].unique()
        col_3_one_hot = np.eye(len(col_3)).astype(np.int)

        col_3_maping = {}
        for idx, col in enumerate(col_3):
            col_3_maping[col] = col_3_one_hot[idx]
        data["col_3_one_hot"] = data[3].apply(lambda x: col_3_maping[x])
        A = []
        for i in data.index:
            A.append(
                [data.loc[i][0]]
                + list(data.loc[i]["age_one_hot"])
                + list(data.loc[i]["col_2_one_hot"])
                + list(data.loc[i]["col_3_one_hot"])
            )
        user_feat = np.stack(A)

        np.savez_compressed(
            f"{self.dataset_dir}/processed/feature_vec.npz",
            user_feat=user_feat,
            item_feat=item_feat,
        )
        return user_feat, item_feat

    def load_fea_vec(self):
        """Loading feature vectors for users and items.
        1. For items (movies), we use the last 19 fields as feature, which are the genres, 
        with 1 indicateing the movie is of that genre, and 0 indicateing it is not; 
        movies can be in several genres at once.
        
        2. For users, we construct one_hot encoding for age, gender and occupation as their
        feature, where ages are categorized into 8 groups.
        
        Returns:
            user_feat (numpy.ndarray): The first column is the user id, rest column are feat vectors
            item_feat (numpy.ndarray): The first column is the itm id, rest column are feat vectors
        
        """
        if not os.path.exists(self.dataset_dir):
            self.preprocess()
        if not os.path.exists(f"{self.dataset_dir}/processed/feature_vec.npz"):
            self.make_fea_vec()
        print(f"Loading user and item feature vactors for dataset {self.dataset_name}")
        loaded = np.load(f"{self.dataset_dir}/processed/feature_vec.npz")
        return loaded["user_feat"], loaded["item_feat"]


class Movielens_1m(DatasetBase):
    def __init__(self):
        """Movielens 1m

        Movielens 1m dataset.
        """
        super().__init__("ml_1m", url=ML_1M_URL)

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, self.dataset_name, "ratings.dat")
        if not os.path.exists(file_name):
            self.download()

        data = pd.read_table(
            file_name,
            header=None,
            sep="::",
            names=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
        )
        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )


class Movielens_25m(DatasetBase):
    def __init__(self):
        """Movielens 25m

        Movielens 25m dataset.
        """
        super().__init__("ml_25m", url=ML_25M_URL)

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, self.dataset_name, "ratings.csv")
        if not os.path.exists(file_name):
            self.download()

        data = pd.read_table(
            file_name,
            header=None,
            sep=",",
            skiprows=[0],
            names=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
        )
        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

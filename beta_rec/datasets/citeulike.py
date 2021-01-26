import csv
import os

import pandas as pd

from ..datasets.dataset_base import DatasetBase
from ..utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_USER_COL

# Download URL.
CULA_URL = "https://github.com/js05212/citeulike-a"
CULT_URL = "https://github.com/js05212/citeulike-t"

# processed data url
CULA_LEAVE_ONE_OUT_URL = "https://1drv.ms/u/s!AjMahLyQeZquggYnM5pZ_sGORKvf?e=oHgSbo"
CULA_RANDOM_SPLIT_URL = "https://1drv.ms/u/s!AjMahLyQeZqugghhNR4XWzUiS501?e=zmVqcx"
CULT_LEAVE_ONE_OUT_URL = "https://1drv.ms/u/s!AjMahLyQeZquggwTOwFEVQojKdyR?e=tTv3DX"
CULT_RANDOM_SPLIT_URL = "https://1drv.ms/u/s!AjMahLyQeZqugg4Ncblkn_gPRxtu?e=YQwM2D"

# Tips
CITEULIKEA_TIPS = """
    CiteULikeA dataset can not be downloaded by this url automatically, and you need to do:
    1. Download this dataset via 'https://github.com/js05212/citeulike-a',
    2. Put 'users.dat' into the directory `citeulike-a/raw/citeulike-a`,
    3. Rename 'users.dat' to 'citeulike_a.dat'
    4. Rerun this program.
"""

CITEULIKET_TIPS = """
    CiteULikeT dataset can not be downloaded by this url automatically, and you need to do:
    1. Download this dataset via 'https://github.com/js05212/citeulike-t',
    2. Put 'users.dat' into the directory `citeulike-t/raw/citeulike-t`,
    3. Rename 'users.dat' to 'citeulike_t.dat'
    4. Rerun this program.
"""


class CiteULikeA(DatasetBase):
    r"""CiteULike-A.

    CiteULike-A dataset. The dataset can not be download by the url, you need to down the dataset by
    'https://github.com/js05212/citeulike-a', then put it into the directory `citeulike-a/raw`
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init CiteULikeA Class."""
        super().__init__(
            "citeulike-a",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            manual_download_url=CULA_URL,
            processed_leave_one_out_url=CULA_LEAVE_ONE_OUT_URL,
            processed_random_split_url=CULA_RANDOM_SPLIT_URL,
            tips=CITEULIKEA_TIPS,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a dataframe consist of the user-item interaction,
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, self.dataset_name, "citeulike_a.dat")
        if not os.path.exists(file_name):
            self.download()

        # Load user-item rating matrix.
        user_item_matrix = pd.read_csv(
            file_name,
            header=None,
            encoding="utf-8",
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )

        # Split each line in user_item_matrix
        userList = []
        itemList = []
        for index, item in user_item_matrix.iterrows():
            rating_list = item[0]
            rating_array = rating_list.split(" ")
            user_id = rating_array[0]
            for i in range(1, len(rating_array)):
                userList.append(user_id)
                itemList.append(rating_array[i])
        prior_transactions = pd.DataFrame({"userID": userList, "itemID": itemList})
        prior_transactions["userID"] = prior_transactions["userID"].astype("int")
        prior_transactions["itemID"] = prior_transactions["itemID"].astype("int")

        # Add rating list into this array
        prior_transactions.insert(2, "rating", 1.0)

        # Rename dataset's columns to fit the standard.
        # Note: there is no timestamp data in this dataset.
        prior_transactions.rename(
            columns={
                "userID": DEFAULT_USER_COL,
                "itemID": DEFAULT_ITEM_COL,
                "rating": DEFAULT_RATING_COL,
            },
            inplace=True,
        )

        # Check the validation of this table.
        print(prior_transactions.head())

        # Save this table.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")


class CiteULikeT(DatasetBase):
    """CiteULike-T.

    CiteULike-T dataset. The dataset can not be download by the url, you need to down the dataset by
    'https://github.com/js05212/citeulike-t', and then put it into the directory `citeulike-t/raw/citeulike-t`.
    """

    def __init__(
        self,
        dataset_name="citeulike-t",
        min_u_c=0,
        min_i_c=3,
    ):
        r"""Init CiteULikeT Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            manual_download_url=CULT_URL,
            processed_leave_one_out_url=CULT_LEAVE_ONE_OUT_URL,
            processed_random_split_url=CULT_RANDOM_SPLIT_URL,
            tips=CITEULIKET_TIPS,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, self.dataset_name, "citeulike_t.dat")
        if not os.path.exists(file_name):
            self.download()

        # Load user-item rating matrix.
        user_item_matrix = pd.read_csv(
            file_name,
            header=None,
            encoding="utf-8",
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )

        # Split each line in user_item_matrix
        userList = []
        itemList = []
        for index, item in user_item_matrix.iterrows():
            rating_list = item[0]
            rating_array = rating_list.split(" ")
            user_id = rating_array[0]
            for i in range(1, len(rating_array)):
                userList.append(user_id)
                itemList.append(rating_array[i])
        prior_transactions = pd.DataFrame({"userID": userList, "itemID": itemList})
        prior_transactions["userID"] = prior_transactions["userID"].astype("int")
        prior_transactions["itemID"] = prior_transactions["itemID"].astype("int")

        # Add rating list into this array
        prior_transactions.insert(2, "rating", 1.0)

        # Rename dataset's columns to fit the standard.
        # Note: there is no timestamp data in this dataset.
        prior_transactions.rename(
            columns={
                "userID": DEFAULT_USER_COL,
                "itemID": DEFAULT_ITEM_COL,
                "rating": DEFAULT_RATING_COL,
            },
            inplace=True,
        )

        # Check the validation of this table.
        print(prior_transactions.head())

        # Save this table.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")

    def load_leave_one_out(
        self, random=False, n_negative=100, n_test=10, download=False
    ):
        r"""Load leave one out split data."""
        if random is False:
            raise RuntimeError(
                "CiteULikeT doesn't have timestamp column, please use random=True as parameter"
            )

        self.load_leave_one_out(random, n_negative, n_test)

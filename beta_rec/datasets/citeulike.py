import os
import csv
import pandas as pd
from beta_rec.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL
from beta_rec.datasets.dataset_base import DatasetBase

# Download URL.
CULA_URL = "https://github.com/js05212/citeulike-a"
CULT_URL = "https://github.com/js05212/citeulike-t"


class CiteULikeA(DatasetBase):
    def __init__(self):
        """CiteULike-A

        CiteULike-A dataset.
        The dataset can not be download by the url,
        you need to down the dataset by 'https://github.com/js05212/citeulike-a'
        then put it into the directory `citeulike-a/raw`
        """
        super().__init__(
            'citeulike-a',
            url=CULA_URL,
            processed_leave_one_out_url="",
            processed_random_split_url="",
            processed_temporal_split_url="",
        )

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = os.path.join(self.raw_path, self.dataset_name, "users.dat")
        if not os.path.exists(file_name):
            self.download()

        # Load user-item rating matrix.
        user_item_matrix = pd.read_csv(
            file_name,
            header=None,
            encoding="utf-8",
            delimiter='\t',
            quoting=csv.QUOTE_NONE
        )

        # Split each line in user_item_matrix
        userList = []
        itemList = []
        for index, item in user_item_matrix.iterrows():
            rating_list = item[0]
            rating_array = rating_list.split(' ')
            user_id = rating_array[0]
            for i in range(1, len(rating_array)):
                userList.append(user_id)
                itemList.append(rating_array[i])
        prior_transactions = pd.DataFrame({"userID": userList, "itemID": itemList})
        prior_transactions["userID"] = prior_transactions["userID"].astype("int")
        prior_transactions["itemID"] = prior_transactions["itemID"].astype("int")

        # Add rating list into this array
        prior_transactions.insert(2, 'rating', 1.0)

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
        # print(prior_transactions.head())

        # Save this table.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")


class CiteULikeT(DatasetBase):
    def __init__(self):
        """CiteULike-T

        CiteULike-T dataset.
        The dataset can not be download by the url,
        you need to down the dataset by 'https://github.com/js05212/citeulike-t'
        then put it into the directory `citeulike-t/raw`
        """
        super().__init__(
            'citeulike-t',
            url=CULT_URL,
            processed_leave_one_out_url="",
            processed_random_split_url="",
            processed_temporal_split_url="",
        )

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = os.path.join(self.raw_path, self.dataset_name, "users.dat")
        if not os.path.exists(file_name):
            self.download()

        # Load user-item rating matrix.
        user_item_matrix = pd.read_csv(
            file_name,
            header=None,
            encoding="utf-8",
            delimiter='\t',
            quoting=csv.QUOTE_NONE
        )

        # Split each line in user_item_matrix
        userList = []
        itemList = []
        for index, item in user_item_matrix.iterrows():
            rating_list = item[0]
            rating_array = rating_list.split(' ')
            user_id = rating_array[0]
            for i in range(1, len(rating_array)):
                userList.append(user_id)
                itemList.append(rating_array[i])
        prior_transactions = pd.DataFrame({"userID": userList, "itemID": itemList})
        prior_transactions["userID"] = prior_transactions["userID"].astype("int")
        prior_transactions["itemID"] = prior_transactions["itemID"].astype("int")

        # Add rating list into this array
        prior_transactions.insert(2, 'rating', 1.0)

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
        # print(prior_transactions.head())

        # Save this table.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")

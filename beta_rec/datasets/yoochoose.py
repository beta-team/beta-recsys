import csv
import os

import pandas as pd

from ..datasets.dataset_base import DatasetBase
from ..utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)

# Download URL
YOOCHOOSE_URL = "https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z"


class YooChoose(DatasetBase):
    """YooChoose Dataset.

    Task of YooChoose dataset: given a sequence of click events performed by
    some users during a typical session in an e-commerce website, the goal is
    to predict whether the user is going to buy something or not, and if he is
    buying, what would be the items he is going to buy. The task could therefore
    be divided into two sub goals:
        1. Is the user going to buy items in this session? YES|NO
        2. If yes, what are the items that are going to be bought?
    This dataset contains two subsets:
        1. yoochoose-clicks.dat
            - SessionID: the id of the session.
            - Timestamp: the time when the click occurred.
            - ItemID: the unique identifier of the item.
            - Category: the category of the item.
        2. yoochoose-buys.dat
            - SessionID: the id of the session.
            - Timestamp: the time when the click occurred.
            - ItemID: the unique identifier of the item.
            - Price: the price of the item.
            - Quantity: how many of this item were bought.

    If the dataset can not be download by the url,
    you need to down the dataset by the link:
        https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z.
    then put it into the directory `yoochoose/raw` and unzip it.
    """

    def __init__(self, dataset_name="yoochoose", min_u_c=0, min_i_c=3, root_dir=None):
        """Init YooChoose Class."""
        super().__init__(
            dataset_name="yelp",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=YOOCHOOSE_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.

        Download datasets if not existed.
        yoochoose_name: yoochoose-buys.dat

        1. Download gowalla dataset if this dataset is not existed.
        2. Load yoochoose <yoochoose-buy> table from 'yoochoose-buys.dat'.
        3. Rename and save dataset model.
        """
        # Step 1: Download gowalla dataset if this dataset is not existed.
        yoochoose_path = os.path.join(self.raw_path, "yoochoose-buys.dat")
        if not os.path.exists(yoochoose_path):
            self.download()

        # Step 2: Load yoochoose <yoochoose-buy> table from 'yoochoose-buys.dat'.
        prior_transactions = pd.read_table(
            yoochoose_path,
            header=None,
            encoding="utf-8",
            sep=",",
            quoting=csv.QUOTE_NONE,
            usecols=[0, 2, 3],
            names=[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL],
        )
        # Add rating columns into this table.
        prior_transactions.insert(3, "rating", 1.0)

        # Step 3: Rename and save dataset model.
        prior_transactions.rename(
            columns={
                "SessionID": DEFAULT_USER_COL,
                "ItemID": DEFAULT_ITEM_COL,
                "rating": DEFAULT_RATING_COL,
                "Timestamp": DEFAULT_TIMESTAMP_COL,
            },
            inplace=True,
        )

        # Check the validation of this dataset.
        # print(prior_transactions.head())

        # Save data model.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")

import os
import time
import pandas as pd
from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase

# Download URL.
DIGINETICA_URL = "https://cikm2016.cs.iupui.edu/cikm-cup/"


def process_time(standard_time=None):
    """Transform time format "xxxx-xx-xx" into format "xxxx-xx-xx xx-xx-xx".
    Since there is no specific hour-minute-second data, we assume it is 00:00:00.

    Args:
        standard_time: str with format "xxxx-xx-xx".
    Returns:
        timestamp: timestamp data.
    """

    standard_time = standard_time + " 00:00:00"
    dateArr = time.strptime(standard_time, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(dateArr))
    return timestamp


class Diginetica(DatasetBase):
    def __init__(self):
        """Diginetica

        Diginetica dataset.
        This is a dataset provided by DIGINETICA and its partners containing
        anonymized search and browsing logs, product data, anonymized transactions,
        and a large data set of product images. The participants have to predict
        search relevance of products according to the personal shopping, search,
        and browsing preferences of the users. Both 'query-less' and 'query-full'
        sessions are possible. The evaluation is based on click and transaction data.

        The dataset can not be download by the url,
        you need to down the dataset by 'https://cikm2016.cs.iupui.edu/cikm-cup/'
        then put it into the directory `diginetica/raw`,
        then unzip this file and rename the new directory to 'diginetica'.

        Note: you also need unzip files in 'diginetica/raw/diginetica'.
        """
        super().__init__('diginetica', DIGINETICA_URL)

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory

        Download datasets if not existed.
        yoochoose_name: yoochoose-buys.dat

        1. Download gowalla dataset if this dataset is not existed.
        2. Load diginetica <diginetica-item-views> table from 'train-item-views.csv'.
        3. Add rating column and create timestamp column.
        4. Save data model.
        """

        # Step 1: Download diginetica dataset if this dataset is not existed.
        diginetica_path = os.path.join(self.raw_path, self.dataset_name, 'train-item-views.csv')
        if not os.path.exists(diginetica_path):
            self.download()

        # Step 2: Load diginetica <diginetica-item-views> table from 'train-item-views.csv'.
        prior_transactions = pd.read_csv(
            diginetica_path,
            header=0,
            encoding="utf-8",
            engine="python",
            sep=';',
            usecols=[0, 2, 4],
            names=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
            nrows=10,
        )

        # Step 3: Add rating column and create timestamp column.
        # Add rating column into this table.
        prior_transactions.insert(2, "col_rating", 1.0)

        # Create timestamp column.
        prior_transactions[DEFAULT_TIMESTAMP_COL] = prior_transactions[DEFAULT_TIMESTAMP_COL].apply(
            lambda t: process_time(t)
        )

        # Check the validation of this dataset.
        # print(prior_transactions.head())

        # Step 4: Save data model.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")

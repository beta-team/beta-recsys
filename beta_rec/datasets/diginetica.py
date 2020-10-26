import os
import time

import pandas as pd

from ..datasets.dataset_base import DatasetBase
from ..utils.constants import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL

# Download URL.
DIGINETICA_URL = "https://cikm2016.cs.iupui.edu/cikm-cup/"

# Tips
DIGINETICA_TIPS = """
    Diginetica dataset can not be downloaded by this url automatically, and you need to do:
    1. Download this dataset via 'https://cikm2016.cs.iupui.edu/cikm-cup/',
    2. Put 'CIKMCUP2016_Track2_DIGINETICA-20200426T024501Z-001.zip' into the directory `diginetica/raw`,
    3. Unzip 'CIKMCUP2016_Track2_DIGINETICA-20200426T024501Z-001.zip',
    4. Rename dir 'CIKMCUP2016_Track2_DIGINETICA' to 'diginetica',
    5. Enter dir 'diginetica' and unzip 'dataset-train-diginetica.zip',
    6. Rename file 'train-item-views.csv' to 'diginetica.csv'
    7. Rerun this program.
"""


def process_time(standard_time=None):
    """Transform time format "xxxx-xx-xx" into format "xxxx-xx-xx xx-xx-xx".

    If there is no specified hour-minute-second data, we use 00:00:00 as default value.

    Args:
        standard_time: str with format "xxxx-xx-xx".
    Returns:
        timestamp: timestamp data.
    """
    standard_time = standard_time + " 00:00:00"
    date_arr = time.strptime(standard_time, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(date_arr))
    return timestamp


class Diginetica(DatasetBase):
    r"""Diginetica Dataset.

    This is a dataset provided by DIGINETICA and its partners containing anonymized search and browsing logs,
    product data, anonymized transactions, and a large data set of product images. The participants have to
    predict search relevance of products according to the personal shopping, search, and browsing preferences of
    the users. Both 'query-less' and 'query-full' sessions are possible. The evaluation is based on click and
    transaction data.

    The dataset can not be download by the url,
    you need to down the dataset by 'https://cikm2016.cs.iupui.edu/cikm-cup/'
    then put it into the directory `diginetica/raw`,
    then unzip this file and rename the new directory to 'diginetica'.

    Note: you also need unzip files in 'diginetica/raw/diginetica'.
    """

    def __init__(self, dataset_name="diginetica", min_u_c=0, min_i_c=3, root_dir=None):
        """Init Diginetica Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            manual_download_url=DIGINETICA_URL,
            tips=DIGINETICA_TIPS,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory

        Download datasets if not existed.
        diginetica_name: train-item-views.csv

        1. Download diginetica dataset if this dataset is not existed.
        2. Load diginetica <diginetica-item-views> table from 'diginetica.csv'.
        3. Add rating column and create timestamp column.
        4. Save data model.
        """
        # Step 1: Download diginetica dataset if this dataset is not existed.
        diginetica_path = os.path.join(
            self.raw_path, self.dataset_name, "diginetica.csv"
        )
        if not os.path.exists(diginetica_path):
            self.download()

        # Step 2: Load diginetica <diginetica-item-views> table from 'diginetica.csv'.
        prior_transactions = pd.read_csv(
            diginetica_path,
            header=0,
            encoding="utf-8",
            engine="python",
            sep=";",
            usecols=[0, 2, 4],
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
        )

        # Step 3: Add rating column and create timestamp column.
        # Add rating column into this table.
        prior_transactions.insert(2, "col_rating", 1.0)

        # Create timestamp column.
        prior_transactions[DEFAULT_TIMESTAMP_COL] = prior_transactions[
            DEFAULT_TIMESTAMP_COL
        ].apply(lambda t: process_time(t))

        # Check the validation of this dataset.
        print(prior_transactions.head())

        # Step 4: Save data model.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")

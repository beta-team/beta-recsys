import os
import time

import pandas as pd

from ..datasets.dataset_base import DatasetBase
from ..utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)

# Download URL
GOWALLA_CHECKIN_URL = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
GOWALLA_EDGES_URL = "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"

# processed data url
GOWALLA_RANDOM_SPLIT_URL = "https://1drv.ms/u/s!AjMahLyQeZqughJgziqB9ORAzcs5?e=wdHaxf"
GOWALLA_TEMPORAL_SPLIT_UTL = "https://1drv.ms/u/s!AjMahLyQeZqughRfFX6k7Kj58NmI?e=iM5f3S"


def process_time(standard_time=None):
    """Transform time format "xxxx-xx-xxTxx-xx-xxZ" into format "xxxx-xx-xx xx-xx-xx".

    Args:
        standard_time: str with format "xxxx-xx-xxTxx-xx-xxZ".
    Returns:
        timestamp: timestamp data.
    """
    standard_time_list = list(standard_time)
    standard_time_list[10] = " "
    standard_time_list.pop()
    standard_time = "".join(standard_time_list)
    date_arr = time.strptime(standard_time, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(date_arr))
    return timestamp


class Gowalla(DatasetBase):
    """Gowalla Dataset.

    Gowalla is a location-based social networking website where users share
    their locations by checking-in. The friendship network is undirected and
    was collected using their public API, and consists of 196,591 nodes and
    950,327 edges. We have collected a total of 6,442,890 check-ins of these
    users over the period of Feb. 2009 - Oct. 2010.

    If the dataset can not be download by the url,
    you need to down the dataset by the link:
        https://snap.stanford.edu/data/loc-Gowalla.html.
    then put it into the directory `gowalla/raw` and unzip it.
    """

    def __init__(self, dataset_name="gowalla", min_u_c=0, min_i_c=3, root_dir=None):
        """Init Gowalla Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=GOWALLA_CHECKIN_URL,
            processed_random_split_url=GOWALLA_RANDOM_SPLIT_URL,
            processed_temporal_split_url=GOWALLA_TEMPORAL_SPLIT_UTL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory

        Download datasets if not existed.
        Gowalla_checkin_name: Gowalla_totalCheckins.txt
        Gowalla_edges_name  : Gowalla_edges.txt

        1. Download gowalla dataset if this dataset is not existed.
        2. Load gowalla <Gowalla_checkin> table from 'Gowalla_totalCheckins.txt'.
        3. Process time columns and transform it into timestamp.
        4. Rename and save dataset model.
        """
        # Step 1: Download gowalla dataset if this dataset is not existed.
        gowalla_path_checkin = os.path.join(self.raw_path, "Gowalla_totalCheckins.txt")
        gowalla_path_edges = os.path.join(self.raw_path, "Gowalla_edges.txt")
        if not os.path.exists(gowalla_path_checkin):
            self.download()
            self.url = GOWALLA_EDGES_URL
        if not os.path.exists(gowalla_path_edges):
            self.download()

        # Step 2: Load gowalla <Gowalla_checkin> table from 'Gowalla_totalCheckins.txt'
        prior_transactions = pd.read_table(
            gowalla_path_checkin,
            header=None,
            sep="\t",
            usecols=[0, 1, 4],
            names=[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL],
        )
        # Add rating column into the table.
        prior_transactions.insert(2, "rating", 1.0)

        # Step 3: Process time columns and transform it into timestamp.
        prior_transactions[DEFAULT_TIMESTAMP_COL] = prior_transactions[
            DEFAULT_TIMESTAMP_COL
        ].apply(lambda t: process_time(t))

        # Step 4: Rename and save dataset model.
        prior_transactions.rename(
            columns={
                DEFAULT_USER_COL: DEFAULT_USER_COL,
                DEFAULT_ITEM_COL: DEFAULT_ITEM_COL,
                "rating": DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL: DEFAULT_TIMESTAMP_COL,
            },
            inplace=True,
        )

        # Check the validation of this table.
        print(prior_transactions.head())

        # Save data model.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")

import os

import pandas as pd

from ..datasets.dataset_base import DatasetBase
from ..utils.constants import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL

# Download URL
RETAIL_ROCKET_URL = "https://www.kaggle.com/retailrocket/ecommerce-dataset/download"

# Tips
RETAIL_ROCKET_TIPS = """
    RetailRocket dataset can not be downloaded by this url automatically, and you need to do:
    1. Download this dataset via 'https://www.kaggle.com/retailrocket/ecommerce-dataset/download',
    2. Put 'ecommerce-dataset.zip' into the directory `retailrocket/raw`,
    3. Unzip 'ecommerce-dataset.zip',
    4. Rerun this program.
"""


class RetailRocket(DatasetBase):
    """RetailRocket Dataset.

    This data has been collected from a real-world e-commerce website. It is
    raw data without any content transformations, however, all values are
    hashed due to confidential issue. The purpose of publishing is to motivate
    researches in the field of recommendation systems with implicit feedback.

    If the dataset can not be download by the url,
    you need to down the dataset by the link:
        https://www.kaggle.com/retailrocket/ecommerce-dataset/download.
    then put it into the directory `retailrocket/raw` and unzip it.
    """

    def __init__(
        self, dataset_name="retailrocket", min_u_c=0, min_i_c=3, root_dir=None
    ):
        """Init RetailRocket Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            manual_download_url=RETAIL_ROCKET_URL,
            tips=RETAIL_ROCKET_TIPS,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a DataFrame consist of the user-item interaction
        and save in the processed directory.

        Download dataset if not existed.
        retail_rocket_name: UserBehavior.csv

        1. Download RetailRocket dataset if this dataset is not existed.
        2. Load RetailRocket <retail-rocket-interaction> table from 'events.csv'.
        3. Save dataset model.
        """
        # Step 1: Download RetailRocket dataset if this dataset is not existed.
        retail_rocket_path = os.path.join(self.raw_path, "events.csv")
        if not os.path.exists(retail_rocket_path):
            self.download()

        # Step 2: Load RetailRocket <retail-rocket-interaction> table from 'events.csv'.
        prior_transactions = pd.read_csv(
            retail_rocket_path,
            engine="python",
            encoding="utf-8",
            header=0,
            usecols=[0, 1, 3],
            names=[DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL, DEFAULT_ITEM_COL],
        )
        # Add rating column into the table.
        prior_transactions.insert(2, "col_rating", 1.0)

        # Step 3: Save dataset model.
        # Check the validation of this dataset.
        print(prior_transactions.head())

        # Save this data model.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")

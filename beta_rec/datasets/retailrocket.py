import os
import pandas as pd
from beta_rec.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL
from beta_rec.datasets.dataset_base import DatasetBase

# Download URL
RETAIL_ROCKET_URL = "https://www.kaggle.com/retailrocket/ecommerce-dataset/download"


class RetailRocket(DatasetBase):
    def __init__(self):
        """RetailRocket

        RetailRocket dataset.
        This data has been collected from a real-world e-commerce website. It is
        raw data without any content transformations, however, all values are
        hashed due to confidential issue. The purpose of publishing is to motivate
        researches in the field of recommendation systems with implicit feedback.

        If the dataset can not be download by the url,
        you need to down the dataset by the link:
            https://www.kaggle.com/retailrocket/ecommerce-dataset/download.
        then put it into the directory `retailrocket/raw` and unzip it.
        """
        super().__init__("retailrocket", RETAIL_ROCKET_URL)

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory

        Download datasets if not existed.
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
            names=[
                DEFAULT_TIMESTAMP_COL,
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL
            ],
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

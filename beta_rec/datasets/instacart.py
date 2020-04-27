import os
import datetime
import numpy as np
import pandas as pd
from beta_rec.utils.common_util import un_zip, timeit
from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase

# Download URL.
INSTACART_URL = 'https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz'

# processed data url
INSTACART_RANDOM_SPLIT_URL = r'https://1drv.ms/u/s!AjMahLyQeZqugX4W4zLO6Jkx8P-W?e=oKymnV'
INSTACART_TEMPORAL_SPLIT_URL = r'https://1drv.ms/u/s!AjMahLyQeZquggAblxVFSYeu3nzh?e=pzBaAa'


class Instacart(DatasetBase):
    def __init__(self):
        """Instacart

        Instacart dataset
        If the dataset can not be download by the url,
        you need to down the dataset by the link:
            'https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz'
        then put it into the directory `instacart/raw`, unzip this file and rename the directory in 'instacart'.

        Instacart dataset is used to predict when users buy
        product for the next time, we construct it with structure [order_id, product_id] => 
        """
        super().__init__(
            'instacart',
            url=INSTACART_URL,
            processed_random_split_url=INSTACART_RANDOM_SPLIT_URL,
            processed_temporal_split_url=INSTACART_TEMPORAL_SPLIT_URL,
        )

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory

        Download and load datasets
        1. Download instacart dataset if this dataset is not existed.
        2. Load <order> table and <order_products> table from "orders.csv" and "order_products__train.csv".
        3. Merge the two tables above.
        4. Add additional columns [rating, timestamp].
        5. Rename columns and save data model.
        """

        # Step 1: Download instacart dataset if this dataset is not existed.
        order_file_path = os.path.join(self.raw_path, self.dataset_name, "orders.csv")
        order_product_file_path = os.path.join(self.raw_path, self.dataset_name, "order_products__train.csv")
        if not os.path.exists(order_file_path) or not os.path.exists(order_product_file_path):
            print("Raw file doesn't exist, try to download it.")
            self.download()

        # Step 2: Load <order> table and <order_products> table from "orders.csv" and "order_products__train.csv".
        path_name = os.path.join(self.raw_path, self.dataset_name)
        orders_data = os.path.join(path_name, "orders.csv")
        orders_products_data = os.path.join(path_name, "order_products__train.csv")

        # orders_table
        orders_table = pd.read_csv(
            orders_data,
            usecols=[
                "order_id", 
                "user_id",
                "order_hour_of_day",
                "days_since_prior_order"
            ]
        )

        # orders_products_table
        orders_products_table = pd.read_csv(
            orders_products_data,
            usecols=[
                "order_id", 
                "product_id",
            ],
        )

        # Step 3: Merge the two tables above.
        prior_transactions = orders_products_table.merge(
            orders_table,
            left_on="order_id",
            right_on="order_id",
        )

        # Step 5: Add additional columns [rating, timestamp].
        # Add rating columns into table.
        prior_transactions.insert(3, "rating", 1)

        # Create virtual timestamp and add to the merge table.
        virtual_date = datetime.datetime.strptime("2020-04-21 00:00", "%Y-%m-%d %H:%M")
        prior_transactions["days_since_prior_order"] = prior_transactions["days_since_prior_order"].apply(
            lambda t: (virtual_date + datetime.timedelta(days=t)).timestamp())
        prior_transactions = prior_transactions.drop(["order_hour_of_day"], axis=1)

        # Step 6: Rename columns and save data model.
        prior_transactions.rename(
            columns = {
                "user_id": DEFAULT_USER_COL,
                "order_id": DEFAULT_ORDER_COL,
                "product_id": DEFAULT_ITEM_COL,
                "rating": DEFAULT_RATING_COL,
                "days_since_prior_order": DEFAULT_TIMESTAMP_COL,
            },
            inplace=True,
        )

        # Check the validation of this table.
        # print(prior_transactions.head(10))

        # save processed data into the disk.
        self.save_dataframe_as_npz(prior_transactions,
                                   os.path.join(self.processed_path, f'{self.dataset_name}_interaction.npz'))

        print("Done.")

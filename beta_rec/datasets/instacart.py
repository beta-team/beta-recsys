import os
import random

import pandas as pd

from ..datasets.dataset_base import DatasetBase
from ..utils.common_util import un_zip
from ..utils.constants import (
    DEFAULT_FLAG_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_ORDER_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)

# Download URL.
INSTACART_URL = "https://www.kaggle.com/c/instacart-market-basket-analysis/data"

# processed data url
INSTACART_RANDOM_SPLIT_URL = (
    r"https://1drv.ms/u/s!AjMahLyQeZqugX4W4zLO6Jkx8P-W?e=oKymnV"
)
INSTACART_TEMPORAL_SPLIT_URL = (
    r"https://1drv.ms/u/s!AjMahLyQeZquggAblxVFSYeu3nzh?e=pzBaAa"
)
INSTACART_LEAVE_ONE_OUT_URL = (
    r"https://1drv.ms/u/s!AjMahLyQeZquggLQynzcCWfNUdIg?e=HDhUjL"
)

# Tips
INSTACART_TIPS = """
    Instacart dataset can not be downloaded by this url automatically, and you need to do:
    1. Download this dataset via 'https://www.kaggle.com/c/instacart-market-basket-analysis/data',
    2. Put 'instacart-market-basket-analysis.zip' into the directory `instacart/raw/`,
    3. Unzip 'instacart-market-basket-analysis.zip', put all the *.csv files into 'instacart/raw/'.
    4. Rerun this program.
"""


class Instacart(DatasetBase):
    """Instacart Dataset.

    If the dataset can not be download by the url,
    you need to down the dataset by the link:
        'https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz'
    then put it into the directory `instacart/raw`, unzip this file and rename the directory in 'instacart'.

    Instacart dataset is used to predict when users buy
    product for the next time, we construct it with structure [order_id, product_id].
    """

    def __init__(
        self, dataset_name="instacart", min_u_c=0, min_i_c=3, min_o_c=0, root_dir=None
    ):
        """Init Instacart Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            min_o_c=min_o_c,
            root_dir=root_dir,
            manual_download_url=INSTACART_URL,
            processed_leave_one_out_url=INSTACART_LEAVE_ONE_OUT_URL,
            processed_random_split_url=INSTACART_RANDOM_SPLIT_URL,
            processed_temporal_split_url=INSTACART_TEMPORAL_SPLIT_URL,
            tips=INSTACART_TIPS,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.

        Download and load datasets
        1. Download instacart dataset if this dataset is not existed.
        2. Load <order> table and <order_products> table from "orders.csv" and "order_products__train.csv".
        3. Merge the two tables above.
        4. Add additional columns [rating, timestamp].
        5. Rename columns and save data model.
        """
        # Step 1: Download instacart dataset if this dataset is not existed.

        print("Start loading data from raw data")
        order_products_prior_file = os.path.join(
            self.raw_path, "order_products__prior.csv"
        )
        order_products_train_file = os.path.join(
            self.raw_path, "order_products__train.csv"
        )
        if not os.path.exists(order_products_prior_file) or not os.path.exists(
            order_products_train_file
        ):
            print("Raw file doesn't exist, try to download it.")
            self.download()

        orders_file = os.path.join(self.raw_path, "orders.csv")

        #  order_products__*.csv: order_id,product_id,add_to_cart_order,reordered
        prior_products = pd.read_csv(
            order_products_prior_file,
            usecols=["order_id", "product_id", "add_to_cart_order"],
        )
        train_products = pd.read_csv(
            order_products_train_file,
            usecols=["order_id", "product_id", "add_to_cart_order"],
        )
        order_products = pd.concat([prior_products, train_products])

        #  orders.csv:  order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order
        orders = pd.read_csv(
            orders_file, usecols=["user_id", "order_id", "order_number", "eval_set"]
        )

        user_products = order_products.merge(orders, how="left", on="order_id")

        # user_item_id = user_products.groupby(["user_id"]).count()

        # user_order_number = user_products.groupby(["user_id", "order_number"]).count()

        order_addtocart_user = (
            user_products.groupby(
                ["order_id", "add_to_cart_order", "user_id", "product_id", "eval_set"]
            )
            .size()
            .rename("ratings")
            .reset_index()
        )
        order_addtocart_user.rename(
            columns={
                "order_id": DEFAULT_ORDER_COL,
                "user_id": DEFAULT_USER_COL,
                "product_id": DEFAULT_ITEM_COL,
                "ratings": DEFAULT_RATING_COL,
                "eval_set": DEFAULT_FLAG_COL,
            },
            inplace=True,
        )
        timestamp_col = {DEFAULT_TIMESTAMP_COL: order_addtocart_user.index}
        order_addtocart_user = order_addtocart_user.assign(**timestamp_col)
        print("Loading raw data completed")
        # save processed data into the disk.
        self.save_dataframe_as_npz(
            order_addtocart_user,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )


class Instacart_25(DatasetBase):
    """Instacart Dataset.

    If the dataset can not be download by the url,
    you need to down the dataset by the link:
        'https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz'
    then put it into the directory `instacart/raw`, unzip this file and rename the directory in 'instacart'.

    Instacart dataset is used to predict when users buy
    product for the next time, we construct it with structure [order_id, product_id].
    """

    def __init__(
        self,
        dataset_name="instacart_25",
        min_u_c=0,
        min_i_c=3,
        min_o_c=0,
    ):
        """Init Instacart_25 Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            min_o_c=min_o_c,
            manual_download_url="https://www.kaggle.com/c/6644/download-all",
            processed_random_split_url=INSTACART_RANDOM_SPLIT_URL,
            processed_temporal_split_url=INSTACART_TEMPORAL_SPLIT_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

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

        print("Start loading data from raw data")
        order_products_prior_file = os.path.join(
            self.raw_path, "order_products__prior.csv"
        )
        order_products_train_file = os.path.join(
            self.raw_path, "order_products__train.csv"
        )
        if not os.path.exists(order_products_prior_file) or not os.path.exists(
            order_products_train_file
        ):
            print("Raw file doesn't exist, try to download it.")
            self.download()
            file_name = os.path.join(self.raw_path + ".gz")
            un_zip(file_name)

        orders_file = os.path.join(self.raw_path, "orders.csv")

        #  order_products__*.csv: order_id,product_id,add_to_cart_order,reordered
        prior_products = pd.read_csv(
            order_products_prior_file,
            usecols=["order_id", "product_id", "add_to_cart_order"],
        )
        train_products = pd.read_csv(
            order_products_train_file,
            usecols=["order_id", "product_id", "add_to_cart_order"],
        )
        order_products = pd.concat([prior_products, train_products])

        #  orders.csv:  order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order
        orders = pd.read_csv(
            orders_file, usecols=["user_id", "order_id", "order_number", "eval_set"]
        )

        user_products = order_products.merge(orders, how="left", on="order_id")

        order_addtocart_user = (
            user_products.groupby(
                ["order_id", "add_to_cart_order", "user_id", "product_id", "eval_set"]
            )
            .size()
            .rename("ratings")
            .reset_index()
        )
        order_addtocart_user.rename(
            columns={
                "order_id": DEFAULT_ORDER_COL,
                "user_id": DEFAULT_USER_COL,
                "product_id": DEFAULT_ITEM_COL,
                "ratings": DEFAULT_RATING_COL,
                "eval_set": DEFAULT_FLAG_COL,
            },
            inplace=True,
        )
        timestamp_col = {DEFAULT_TIMESTAMP_COL: order_addtocart_user.index}
        order_addtocart_user = order_addtocart_user.assign(**timestamp_col)
        print("Start sampling 25% users from the raw data")
        users = list(order_addtocart_user[DEFAULT_USER_COL].unique())
        sampled_users = random.sample(users, int(len(users) * 0.25))
        order_addtocart_user = order_addtocart_user[
            order_addtocart_user[DEFAULT_USER_COL].isin(sampled_users)
        ]

        print("Loading raw data completed")
        # save processed data into the disk.
        self.save_dataframe_as_npz(
            order_addtocart_user,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

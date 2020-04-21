import os
import datetime
import numpy as np
import pandas as pd
from beta_rec.utils.common_util import un_zip, timeit
from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase


class Instacart(DatasetBase):
    def __init__(self):
        """Instacart

        Instacart dataset
        If the dataset can not be download by the url,
        you need to down the dataset by the link:
            'https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz'
        then put it into the directory `instacart/raw`

        Instacart dataset is used to predict when users buy
        product for the next time, we construct it with structure [order_id, product_id] => 
        """
        super().__init__(
            'instacart',
            url = 'https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz'
        )

    
    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        
        # If raw file doesn't exist, download it into your database.
        file_zip_name = os.path.join(self.raw_path, "instacart.zip")
        if not os.path.exists(file_zip_name):
            print("Raw file doesn't exist, try to download it.")
            self.download()

        # Process raw file
        path_name = os.path.join(self.raw_path, self.dataset_name)  
        # orders_table, containing order_id, user_id, and time.
        # orders_products_table, containing product_id, order_id, and reordered info.
        orders_data = os.path.join(path_name, "orders.csv")
        orders_products_data = os.path.join(path_name, "order_products__train.csv")
        products_data = os.path.join(path_name, "products.csv")

        # orders_table
        orders_table = pd.read_csv(
            orders_data,
            usecols=[
                "order_id", 
                "user_id", 
                #"eval_set",
                #"order_number",
                #"order_dow",
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
                #"add_to_cart_order", 
                #"reordered"
            ],
        )

        # products_table
        # Table with info structure [product_id, aisle_id, department_id].
        # products_table = pd.read_csv(
        #     products_data,
        #     usecols=[
        #         "product_id", 
        #         "aisle_id", 
        #         "department_id"
        #     ],
        # )

        # Merge all the four datasets according to order_id, products_id.
        prior_transactions = orders_products_table.merge(
            orders_table,
            left_on = "order_id",
            right_on = "order_id",
        )

        # Merge the products_table
        # prior_transactions = prior_transactions.merge(
        #     products_table,
        #     left_on = "product_id",
        #     right_on = "product_id",
        # )

        # Add rating columns into table.
        prior_transactions.insert(3, "rating", 1)

        # Process time datas
        virtual_date = datetime.datetime.strptime("2020-04-21 00:00", "%Y-%m-%d %H:%M")
        prior_transactions["days_since_prior_order"] = prior_transactions["days_since_prior_order"].apply(lambda t: (virtual_date + datetime.timedelta(days=t)).timestamp())
        prior_transactions = prior_transactions.drop(["order_hour_of_day"], axis=1)

        # Standardize the columns' name.
        prior_transactions.rename(
            columns = {
                "user_id"                : DEFAULT_USER_COL,
                "order_id"               : DEFAULT_ORDER_COL,
                "product_id"             : DEFAULT_ITEM_COL,
                "rating"                 : DEFAULT_RATING_COL, 
                #"aisle_id"              : "aisle_id",
                #"department_id"         : "department_id",
                #"add_to_cart_order"     : "add_to_cart_order",
                #"reordered"             : "reordered",
                #"order_hour_of_day"     : "order_hour_of_day",
                "days_since_prior_order" : DEFAULT_TIMESTAMP_COL,
                #"time"                  : DEFAULT_TIMESTAMP_COL,
            },
            inplace = True,
        )

        # Check the validation of this table.
        # print(prior_transactions.head(10))

        # save processed data into the disk.
        self.save_dataframe_as_npz(prior_transactions, os.path.join(self.processed_path, f'{self.dataset_name}_interaction.npz'))
        
        print("Done.")

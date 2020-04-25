import os
import numpy as np
import pandas as pd
from beta_rec.utils.common_util import un_zip, timeit
from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase

# download_url
DUNNHUMBY_URL = r'https://www.dunnhumby.com/sites/default/files/sourcefiles/dunnhumby_The-Complete-Journey.zip'

# processed data url
DUNNHUMBY_LEAVE_ONE_BASKET_URL = r'https://1drv.ms/u/s!AjMahLyQeZqugXCn99mGZw4uHaSg?e=GhmyCa'
DUNNHUMBY_LEAVE_ONE_OUT_URL = r'https://1drv.ms/u/s!AjMahLyQeZqugXK8xN12i0O4K-dd?e=OG0Dl3'
DUNNHUMBY_RANDOM_SPLIT_URL = r'https://1drv.ms/u/s!AjMahLyQeZqugXRLlZbQnYJbjY1d?e=aQ9LrF'
DUNNHUMBY_RANDOM_BASKET_SPLIT_URL = r'https://1drv.ms/u/s!AjMahLyQeZqugXYbw7U3_M363CpM?e=DuyT3a'
DUNNHUMBY_TEMPORAL_SPLIT_URL = r'https://1drv.ms/u/s!AjMahLyQeZqugXgd1VE2sX089Udc?e=S2eM7Q'
DUNNHUMBY_TEMPORAL_BASKET_SPLIT_URL = r'https://1drv.ms/u/s!AjMahLyQeZqugXrmhlEvrEzYiX42?e=1RNidC'


class Dunnhumby(DatasetBase):
    def __init__(self):
        """Dunnhumby

        Dunnhumby dataset.
        If the dataset can not be download by the url,
        you need to down the dataset by the link:
            'https://www.dunnhumby.com/sites/default/files/sourcefiles/dunnhumby_The-Complete-Journey.zip'
        then put it into the directory `dunnhumby/raw`
        """
        super().__init__(
            "dunnhumby",
            url=DUNNHUMBY_URLp,
            processed_leave_one_basket_url=DUNNHUMBY_LEAVE_ONE_BASKET_URL,
            processed_leave_one_out_url=DUNNHUMBY_LEAVE_ONE_OUT_URL,
            processed_random_split_url=DUNNHUMBY_RANDOM_SPLIT_URL,
            processed_random_basket_split_url=DUNNHUMBY_RANDOM_BASKET_SPLIT_URL,
            processed_temporal_split_url=DUNNHUMBY_TEMPORAL_SPLIT_URL,
            processed_temporal_basket_split_url=DUNNHUMBY_TEMPORAL_BASKET_SPLIT_URL,
        )
        self.load_temporal_split = self.load_temporal_basket_split

    @timeit
    def parse_raw_data(self, data_base_dir="./dunnhumby_The-Complete-Journey"):
        """Parse raw dunnhumby csv data from transaction_data.csv

        Args:
            data_base_dir (path): Default dir is "./dunnhumby - The Complete Journey CSV"

        Returns:
            DataFrame of interactions
        """

        transaction_data = os.path.join(data_base_dir, "transaction_data.csv")
        prior_transaction = pd.read_csv(
            transaction_data,
            usecols=["BASKET_ID", "household_key", "PRODUCT_ID", "DAY", "TRANS_TIME"],
        )

        prior_transaction["DAY"] = prior_transaction["DAY"].astype(str)  #
        prior_transaction["TRANS_TIME"] = prior_transaction["TRANS_TIME"].astype(str)

        prior_transaction["time"] = (
            prior_transaction["DAY"] + prior_transaction["TRANS_TIME"]
        )
        prior_transaction["time"] = prior_transaction["time"].astype(int)  #
        prior_transaction.reset_index(inplace=True)
        prior_transaction = prior_transaction.sort_values(by="time", ascending=False)

        prior_transaction.drop(["DAY", "TRANS_TIME"], axis=1)

        prior_transaction = prior_transaction[
            ["BASKET_ID", "household_key", "PRODUCT_ID", "time"]
        ]
        prior_transaction.insert(3, "flag", "train")
        prior_transaction.insert(4, "ratings", 1)
        prior_transaction.rename(
            columns={
                "BASKET_ID": DEFAULT_ORDER_COL,
                "household_key": DEFAULT_USER_COL,
                "PRODUCT_ID": DEFAULT_ITEM_COL,
                "flag": DEFAULT_FLAG_COL,
                "ratings": DEFAULT_RATING_COL,
                "time": DEFAULT_TIMESTAMP_COL,
            },
            inplace=True,
        )

        print("loading raw data completed")
        return prior_transaction

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = os.path.join(self.raw_path, "dunnhumby.zip")
        if not os.path.exists(file_name):
            print("Raw file doesn't exist, try to download it.")
            self.download()
        zip_file_name = os.path.join(self.raw_path, "dunnhumby.zip")
        unzip_file_name = os.path.join(self.raw_path, "dunnhumby_The-Complete-Journey")
        if not os.path.exists(unzip_file_name):
            un_zip(zip_file_name)

        if not os.path.exists(
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz")
        ):
            data = self.parse_raw_data(
                os.path.join(unzip_file_name, "dunnhumby - The Complete Journey CSV")
            )
            self.save_dataframe_as_npz(
                data,
                os.path.join(
                    self.processed_path, f"{self.dataset_name}_interaction.npz"
                ),
            )

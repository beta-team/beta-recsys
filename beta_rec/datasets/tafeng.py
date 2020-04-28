import os
import pandas as pd

from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase

# download_url
TAFENG_URL = r"https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset/download"

# processed data url
TAFENG_LEAVE_ONE_OUT_URL = r'https://1drv.ms/u/s!AjMahLyQeZqugWw1iWQHgI2NNbuM?e=LwEbEc'
TAFENG_RANDOM_SPLIT_URL = r'https://1drv.ms/u/s!AjMahLyQeZqugWbXQ__YWqF9v_7x?e=NjX5VQ'
TAFENG_TEMPORAL_SPLIT_URL = r'https://1drv.ms/u/s!AjMahLyQeZqugWp1Y1JefMXZr0ng?e=OoAgwD'


class Tafeng(DatasetBase):
    def __init__(self):
        """Tafeng

        Tafeng dataset.
        The dataset can not be download by the url,
        you need to down the dataset by 'https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset/download'
        then put it into the directory `tafeng/raw`
        """
        super().__init__(
            "tafeng",
            manual_download_url=TAFENG_URL,
            processed_random_split_url=TAFENG_RANDOM_SPLIT_URL
        )

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = os.path.join(self.raw_path, self.dataset_name, "ta_feng_all_months_merged.csv")
        if not os.path.exists(file_name):
            self.download()
        data = pd.read_table(
            file_name,
            sep=",",
            usecols=[0, 1, 5, 6],
            names=[
                DEFAULT_TIMESTAMP_COL,
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
            ],
            header=0,
        )

        data[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(data[DEFAULT_TIMESTAMP_COL])
        data[DEFAULT_TIMESTAMP_COL] = data[DEFAULT_TIMESTAMP_COL].map(
            lambda x: x.timestamp()
        )

        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

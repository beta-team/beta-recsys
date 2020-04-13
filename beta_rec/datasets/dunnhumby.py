import os
import numpy as np
import pandas as pd
import zipfile

from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase


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
            manual_download_url="https://www.dunnhumby.com/sites/default/files/sourcefiles/dunnhumby_The-Complete-Journey.zip",
        )
        self.load_temporal_split = self.load_temporal_basket_split

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = os.path.join(self.raw_path, "dunnhumby_The-Complete-Journey.zip")
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

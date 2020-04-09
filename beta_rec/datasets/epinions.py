import sys
import os
import pandas as pd

from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase

epinions_url = 'http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2'

class Epinions(DatasetBase):
    def __init__(self):
        """Epinions
        Epinions dataset.
        """
        super().__init__(epinions_url, 'epinions')
    
    def preprocess(self):
        """Preprocess the raw file
        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = os.path.join(self.download_path, 'ratings_data.txt')
        if not os.path.exists(file_name):
            self.download()

        data = pd.read_table(
            file_name,
            header=None,
            sep=' ',
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]
        )
        self.save_dataframe_as_npz(data, self.processed_file_path)

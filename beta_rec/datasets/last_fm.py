import sys
import os
import pandas as pd

from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase

last_fm_url = 'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip'


class LastFM(DatasetBase):
    def __init__(self):
        """Last.FM
        Last.FM dataset.
        """
        super().__init__(last_fm_url, 'last_fm')
    
    def preprocess(self):
        """Preprocess the raw file
        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = os.path.join(self.download_path, 'user_artists.dat')
        if not os.path.exists(file_name):
            self.download()

        data = pd.read_table(
            file_name,
            header=None,
            sep='\s',
            skiprows=[0],
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]
        )
        self.save_dataframe_as_npz(data, self.processed_file_path)

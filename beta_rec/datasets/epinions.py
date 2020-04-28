import os
import pandas as pd

from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase

# download_url
EPINIONS_URL = r'http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2'

# processed data url
EPIONIONS_LEAVE_ONE_OUT_URL = r'https://1drv.ms/u/s!AjMahLyQeZqugWkmLba7PHAHDUVY?e=rnM4WA'
EPIONIONS_RANDOM_SPLIT_URL = r'https://1drv.ms/u/s!AjMahLyQeZqugVvhh9T04Hff7Tev?e=HF7DCH'


class Epinions(DatasetBase):
    def __init__(self):
        """Epinions

        Epinions dataset.
        """
        super().__init__('epinions', url=EPINIONS_URL,
                         processed_leave_one_out_url=EPIONIONS_LEAVE_ONE_OUT_URL,
                         processed_random_split_url=EPIONIONS_RANDOM_SPLIT_URL)

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = os.path.join(self.raw_path, f'{self.dataset_name}.txt')

        if not os.path.exists(file_name):
            self.download()

        data = pd.read_table(
            file_name,
            header=None,
            sep=' ',
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]
        )
        self.save_dataframe_as_npz(data, os.path.join(self.processed_path, f'{self.dataset_name}_interaction.npz'))

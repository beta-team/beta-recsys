import os

import pandas as pd

from ..datasets.dataset_base import DatasetBase
from ..utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_USER_COL

# download_url
LAST_FM_URL = r"http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"

# processed data url
LAST_FM_LEAVE_ONE_OUT_URL = r"https://1drv.ms/u/s!AjMahLyQeZqugWWzMfP30dibhLE5?e=VeW8fM"
LAST_FM_RANDOM_SPLIT_URL = r"https://1drv.ms/u/s!AjMahLyQeZqugV8roce2jec5yVMs?e=7NpTIb"


class LastFM(DatasetBase):
    """LastFM Dataset."""

    def __init__(self, dataset_name="last_fm", min_u_c=0, min_i_c=3, root_dir=None):
        """Init LastFM Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=LAST_FM_URL,
            processed_leave_one_out_url=LAST_FM_LEAVE_ONE_OUT_URL,
            processed_random_split_url=LAST_FM_RANDOM_SPLIT_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a DataFrame consist of the user-item interaction
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, "user_artists.dat")
        if not os.path.exists(file_name):
            self.download()

        data = pd.read_table(
            file_name,
            header=None,
            sep=r"\s",
            skiprows=[0],
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL],
        )
        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

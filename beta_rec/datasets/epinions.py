import os

import pandas as pd

from ..datasets.dataset_base import DatasetBase
from ..utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_USER_COL

# download_url
EPINIONS_URL = (
    r"http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2"
)

# processed data url
EPIONIONS_LEAVE_ONE_OUT_URL = (
    r"https://1drv.ms/u/s!AjMahLyQeZqugWkmLba7PHAHDUVY?e=rnM4WA"
)
EPIONIONS_RANDOM_SPLIT_URL = (
    r"https://1drv.ms/u/s!AjMahLyQeZqugVvhh9T04Hff7Tev?e=HF7DCH"
)


class Epinions(DatasetBase):
    """Epinions Dataset."""

    def __init__(self, dataset_name="epinions", min_u_c=0, min_i_c=3, root_dir=None):
        """Init Epinions Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=EPINIONS_URL,
            processed_leave_one_out_url=EPIONIONS_LEAVE_ONE_OUT_URL,
            processed_random_split_url=EPIONIONS_RANDOM_SPLIT_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, f"{self.dataset_name}.txt")

        if not os.path.exists(file_name):
            self.download()

        data = pd.read_table(
            file_name,
            header=None,
            sep=" ",
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL],
        )
        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

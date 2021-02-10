import csv
import os

import pandas as pd

from ..datasets.dataset_base import DatasetBase
from ..utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_ORDER_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)

# Download URLs
ML_2K_URL = (
    "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip"
)
DL_2K_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip"
LF_2K_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"

# processed data url
ML_2K_LEAVE_ONE_OUT_URL = "https://1drv.ms/u/s!AjMahLyQeZqugjLTDesYUMDz7m4-?e=YjAeFC"
ML_2K_RANDOM_URL = "https://1drv.ms/u/s!AjMahLyQeZqugjQGQk6tnU_abBhJ?e=ewH1dM"
ML_2K_TEMPORAL_URL = "https://1drv.ms/u/s!AjMahLyQeZqugja3hXLGo-74UziC?e=6MN7jh"
DL_2K_LEAVE_ONE_BASKET_URL = "https://1drv.ms/u/s!AjMahLyQeZqugiwhklT9W7jC1jCs?e=aO2IZz"
DL_2K_LEAVE_ONE_OUT_URL = "https://1drv.ms/u/s!AjMahLyQeZqugi6ZpsxBQ-6KbXmE?e=wTyIX7"
LF_2K_LEAVE_ONE_BASKET_URL = "https://1drv.ms/u/s!AjMahLyQeZqugh0S50BjwNBDIVSi?e=ZSgjoB"
LF_2K_LEAVE_ONE_OUT_URL = "https://1drv.ms/u/s!AjMahLyQeZqugh9SOy0tpGT-DyGO?e=djcSUS"
LF_2K_RANDOM_URL = "https://1drv.ms/u/s!AjMahLyQeZqugiE-a65YO3G7ziq4?e=XKla4F"
LF_2K_RANDOM_BASKET_URL = "https://1drv.ms/u/s!AjMahLyQeZqugiM00QUQJYPWA5YE?e=s5o7By"
LF_2K_TEMPORAL_URL = "https://1drv.ms/u/s!AjMahLyQeZqugiWDnPcNDr-5Hyri?e=m8fqQa"
LF_2K_TEMPORAL_BASKET_URL = "https://1drv.ms/u/s!AjMahLyQeZqugic_tmXvkpxN_RE8?e=ZKuSbB"


class MovieLens_2k(DatasetBase):
    """MovieLens-2k Dataset.

    If the dataset can not be download by the url,
    you need to down the dataset by the link:
    'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip'
    then put it into the directory `movielens-2k/raw.
    """

    def __init__(
        self, dataset_name="movielens-2k", min_u_c=0, min_i_c=3, root_dir=None
    ):
        """Init Movielens_2k Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            manual_download_url=ML_2K_URL,
            url=ML_2K_URL,
            processed_leave_one_out_url="",
            processed_random_split_url="",
            processed_temporal_split_url="",
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        movie_2k_file = os.path.join(self.raw_path, "user_ratedmovies-timestamps.dat")
        if not os.path.exists(movie_2k_file):
            self.download()

        # Load in [user, bookmark, tags, timestamps] format.
        prior_transactions = pd.read_csv(
            movie_2k_file,
            header=0,
            encoding="utf-8",
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )

        # Rename this table to fix the standard.
        prior_transactions.rename(
            columns={
                "userID": DEFAULT_USER_COL,
                "movieID": DEFAULT_ITEM_COL,
                "rating": DEFAULT_RATING_COL,
                "timestamp": DEFAULT_TIMESTAMP_COL,
            },
            inplace=True,
        )

        # Check the validation of prior_transactions.
        print(prior_transactions.head())

        # Save data.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")


class Delicious_2k(DatasetBase):
    """delicious-2k Dataset.

    This dataset contains social networking, bookmarking, and tagging information
    from a set of 2K users from Delicious social bookmarking system.
    http://www.delicious.com.

    If the dataset can not be download by the url,
    you need to down the dataset in the following link:
    'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip'
    then put it into the directory `delicious-2k/raw`.
    """

    def __init__(
        self,
        dataset_name="delicious-2k",
        min_u_c=0,
        min_i_c=3,
        root_dir=None,
    ):
        """Init Delicious_2k Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            manual_download_url=DL_2K_URL,
            url=DL_2K_URL,
            processed_leave_one_out_url="",
            processed_random_split_url="",
            processed_temporal_split_url="",
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        delicious_file = os.path.join(
            self.raw_path, "user_taggedbookmarks-timestamps.dat"
        )
        if not os.path.exists(delicious_file):
            self.download()

        # Load in [user, bookmark, tags, timestamps] format.
        prior_transactions = pd.read_csv(
            delicious_file,
            header=0,
            encoding="utf-8",
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )

        # Add rating feature into this table.
        prior_transactions.insert(3, "rating", 1)

        # Rename this table to fix the standard.
        prior_transactions.rename(
            columns={
                "userID": DEFAULT_USER_COL,
                "bookmarkID": DEFAULT_ITEM_COL,
                "tagID": DEFAULT_ORDER_COL,
                "rating": DEFAULT_RATING_COL,
                "timestamp": DEFAULT_TIMESTAMP_COL,
            },
            inplace=True,
        )

        # Check the validation of prior_transactions.
        # print(prior_transactions.head())

        # Save data.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")


class LastFM_2k(DatasetBase):
    """Lastfm-2k Dataset.

    This dataset contains social networking, tagging, and music artist listening information
    from a set of 2K users from Last.fm online music system.

    If the dataset can not be download by the url,
    you need to down the dataset by the link:
        'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip'
    then put it into the directory `delicious-2k/raw`.
    """

    def __init__(self, dataset_name="lastfm-2k", min_u_c=0, min_i_c=3, root_dir=None):
        """Init LastFM_2k Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            manual_download_url=LF_2K_URL,
            url=LF_2K_URL,
            processed_leave_one_basket_url=LF_2K_LEAVE_ONE_BASKET_URL,
            processed_leave_one_out_url=LF_2K_LEAVE_ONE_OUT_URL,
            processed_random_basket_split_url=LF_2K_RANDOM_BASKET_URL,
            processed_random_split_url=LF_2K_RANDOM_URL,
            processed_temporal_basket_split_url=LF_2K_TEMPORAL_BASKET_URL,
            processed_temporal_split_url=LF_2K_TEMPORAL_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        lastfm_file = os.path.join(self.raw_path, "user_taggedartists-timestamps.dat")
        if not os.path.exists(lastfm_file):
            self.download()

        # Load in [user, bookmark, tags, timestamps] format.
        prior_transactions = pd.read_csv(
            lastfm_file,
            header=0,
            encoding="utf-8",
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )

        # Add rating feature into this table.
        prior_transactions.insert(3, "rating", 1)

        # Rename this table to fix the standard.
        prior_transactions.rename(
            columns={
                "userID": DEFAULT_USER_COL,
                "artistID": DEFAULT_ITEM_COL,
                "tagID": DEFAULT_ORDER_COL,
                "rating": DEFAULT_RATING_COL,
                "timestamp": DEFAULT_TIMESTAMP_COL,
            },
            inplace=True,
        )

        # Check the validation of prior_transactions.
        print(prior_transactions.head())

        # Save data.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")

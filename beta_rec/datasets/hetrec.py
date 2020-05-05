import os
import csv
import pandas as pd
from beta_rec.utils.constants import *
from beta_rec.datasets.dataset_base import DatasetBase

# Download URLs
ML_2K_URL = 'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip'
DL_2K_URL = 'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip'
LF_2K_URL = 'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip'


class MovieLens_2k(DatasetBase):
    def __init__(self):
        """MovieLens-2k

        MovieLens-2k dataset
        If the dataset can not be download by the url,
        you need to down the dataset by the link:
            'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip'
        then put it into the directory `movielens-2k/raw
        """

        super().__init__(
            'movielens-2k',
            url=ML_2K_URL,
            processed_leave_one_out_url="",
            processed_random_split_url="",
            processed_temporal_split_url="",
        )

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        movie_2k_file = os.path.join(self.raw_path, 'user_ratedmovies-timestamps.dat')
        if not os.path.exists(movie_2k_file):
            self.download()

        # Load in [user, bookmark, tags, timestamps] format.
        prior_transactions = pd.read_csv(
            movie_2k_file,
            header=0,
            encoding="utf-8",
            delimiter='\t',
            quoting=csv.QUOTE_NONE
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
        # print(prior_transactions.head())

        # Save data.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f'{self.dataset_name}_interaction.npz')
        )

        print("Done.")


class Delicious_2k(DatasetBase):
    def __init__(self):
        """delicious-2k

        Delicious-2k dataset
        This dataset contains social networking, bookmarking, and tagging information
        from a set of 2K users from Delicious social bookmarking system.
        http://www.delicious.com

        If the dataset can not be download by the url,
        you need to down the dataset in the following link:
            'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip'
        then put it into the directory `delicious-2k/raw
        """

        super().__init__(
            'delicious-2k',
            url=DL_2K_URL,
            processed_leave_one_out_url="",
            processed_random_split_url="",
            processed_temporal_split_url="",
        )

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        delicious_file = os.path.join(self.raw_path, 'user_taggedbookmarks-timestamps.dat')
        if not os.path.exists(delicious_file):
            self.download()

        # Load in [user, bookmark, tags, timestamps] format.
        prior_transactions = pd.read_csv(
            delicious_file,
            header=0,
            encoding="utf-8",
            delimiter='\t',
            quoting=csv.QUOTE_NONE
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
            os.path.join(self.processed_path, f'{self.dataset_name}_interaction.npz')
        )

        print("Done.")


class LastFM_2k(DatasetBase):
    def __init__(self):
        """lastfm-2k

        lastfm-2k dataset
        This dataset contains social networking, tagging, and music artist listening information
        from a set of 2K users from Last.fm online music system.

        If the dataset can not be download by the url,
        you need to down the dataset by the link:
            'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip'
        then put it into the directory `delicious-2k/raw
        """

        super().__init__(
            'lastfm-2k',
            url=LF_2K_URL,
            processed_leave_one_out_url="",
            processed_random_split_url="",
            processed_temporal_split_url="",
        )

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        lastfm_file = os.path.join(self.raw_path, 'user_taggedartists-timestamps.dat')
        if not os.path.exists(lastfm_file):
            self.download()

        # Load in [user, bookmark, tags, timestamps] format.
        prior_transactions = pd.read_csv(
            lastfm_file,
            header=0,
            encoding="utf-8",
            delimiter='\t',
            quoting=csv.QUOTE_NONE
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
        # print(prior_transactions.head())

        # Save data.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f'{self.dataset_name}_interaction.npz')
        )

        print("Done.")

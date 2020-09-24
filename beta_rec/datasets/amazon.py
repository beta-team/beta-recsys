import os

import pandas as pd

from beta_rec.datasets.dataset_base import DatasetBase
from beta_rec.utils.common_util import get_data_frame_from_gzip_file
from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)

# Download URL.
AMAZON_Amazon_Instant_Video_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Amazon_Instant_Video.json.gz"
)
AMAZON_Musical_Instruments_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Musical_Instruments.json.gz"
)
AMAZON_Digital_Music_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Digital_Music.json.gz"
)
AMAZON_Baby_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Patio_Lawn_and_Garden.json.gz"
)
AMAZON_Patio_Lawn_Garden_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Patio_Lawn_and_Garden.json.gz"
)
AMAZON_Grocery_Gourmet_Food_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Grocery_and_Gourmet_Food.json.gz"
)
AMAZON_Automotive_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Automotive.json.gz"
)
AMAZON_Pet_Supplies_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Pet_Supplies.json.gz"
)


class AmazonInstantVideo(DatasetBase):
    r"""AmazonInstantVideo.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonInstantVideo Class."""
        super().__init__(
            dataset_name="amazon-amazon-instant-video",
            root_dir=root_dir,
            url=AMAZON_Amazon_Instant_Video_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a dataframe consist of the user-item interaction,
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, "amazon-amazon-instant-video.json.gz")
        print(f"file_name: {file_name}")
        if not os.path.exists(file_name):
            self.download()

        # parse json data
        data = get_data_frame_from_gzip_file(file_name)

        # rename columns
        data = data.rename(
            columns={
                "reviewerID": DEFAULT_USER_COL,
                "asin": DEFAULT_ITEM_COL,
                "overall": DEFAULT_RATING_COL,
                "unixReviewTime": DEFAULT_TIMESTAMP_COL,
            }
        )

        # select necessary columns
        data = pd.DataFrame(
            data,
            columns=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
        )

        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )


class AmazonMusicalInstruments(DatasetBase):
    r"""AmazonMusicalInstruments.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonMusicalInstruments Class."""
        super().__init__(
            dataset_name="amazon-musical-instruments",
            root_dir=root_dir,
            url=AMAZON_Musical_Instruments_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a dataframe consist of the user-item interaction,
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, "amazon-musical-instruments.json.gz")
        print(f"file_name: {file_name}")
        if not os.path.exists(file_name):
            self.download()

        # parse json data
        data = get_data_frame_from_gzip_file(file_name)

        # rename columns
        data = data.rename(
            columns={
                "reviewerID": DEFAULT_USER_COL,
                "asin": DEFAULT_ITEM_COL,
                "overall": DEFAULT_RATING_COL,
                "unixReviewTime": DEFAULT_TIMESTAMP_COL,
            }
        )

        # select necessary columns
        data = pd.DataFrame(
            data,
            columns=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
        )

        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )


class AmazonDigitalMusic(DatasetBase):
    r"""AmazonDigitalMusic.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonDigitalMusic Class."""
        super().__init__(
            dataset_name="amazon-digital-music",
            root_dir=root_dir,
            url=AMAZON_Digital_Music_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a dataframe consist of the user-item interaction,
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, "amazon-digital-music.json.gz")
        print(f"file_name: {file_name}")
        if not os.path.exists(file_name):
            self.download()

        # parse json data
        data = get_data_frame_from_gzip_file(file_name)

        # rename columns
        data = data.rename(
            columns={
                "reviewerID": DEFAULT_USER_COL,
                "asin": DEFAULT_ITEM_COL,
                "overall": DEFAULT_RATING_COL,
                "unixReviewTime": DEFAULT_TIMESTAMP_COL,
            }
        )

        # select necessary columns
        data = pd.DataFrame(
            data,
            columns=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
        )

        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )


class AmazonBaby(DatasetBase):
    r"""AmazonBaby.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonBaby Class."""
        super().__init__(
            dataset_name="amazon-baby", root_dir=root_dir, url=AMAZON_Baby_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a dataframe consist of the user-item interaction,
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, "amazon-baby.json.gz")
        print(f"file_name: {file_name}")
        if not os.path.exists(file_name):
            self.download()

        # parse json data
        data = get_data_frame_from_gzip_file(file_name)

        # rename columns
        data = data.rename(
            columns={
                "reviewerID": DEFAULT_USER_COL,
                "asin": DEFAULT_ITEM_COL,
                "overall": DEFAULT_RATING_COL,
                "unixReviewTime": DEFAULT_TIMESTAMP_COL,
            }
        )

        # select necessary columns
        data = pd.DataFrame(
            data,
            columns=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
        )

        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )


class AmazonPatioLawnGarden(DatasetBase):
    r"""AmazonPatioLawnGarden.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonPatioLawnGarden Class."""
        super().__init__(
            dataset_name="amazon-patio-lawn-garden",
            root_dir=root_dir,
            url=AMAZON_Patio_Lawn_Garden_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a dataframe consist of the user-item interaction,
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, "amazon-patio-lawn-garden.json.gz")
        print(f"file_name: {file_name}")
        if not os.path.exists(file_name):
            self.download()

        # parse json data
        data = get_data_frame_from_gzip_file(file_name)

        # rename columns
        data = data.rename(
            columns={
                "reviewerID": DEFAULT_USER_COL,
                "asin": DEFAULT_ITEM_COL,
                "overall": DEFAULT_RATING_COL,
                "unixReviewTime": DEFAULT_TIMESTAMP_COL,
            }
        )

        # select necessary columns
        data = pd.DataFrame(
            data,
            columns=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
        )

        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )


class AmazonGroceryGourmetFood(DatasetBase):
    r"""AmazonGroceryGourmetFood.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonGroceryGourmetFood Class."""
        super().__init__(
            dataset_name="amazon-grocery-gourmet-food",
            root_dir=root_dir,
            url=AMAZON_Grocery_Gourmet_Food_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a dataframe consist of the user-item interaction,
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, "amazon-grocery-gourmet-food.json.gz")
        print(f"file_name: {file_name}")
        if not os.path.exists(file_name):
            self.download()

        # parse json data
        data = get_data_frame_from_gzip_file(file_name)

        # rename columns
        data = data.rename(
            columns={
                "reviewerID": DEFAULT_USER_COL,
                "asin": DEFAULT_ITEM_COL,
                "overall": DEFAULT_RATING_COL,
                "unixReviewTime": DEFAULT_TIMESTAMP_COL,
            }
        )

        # select necessary columns
        data = pd.DataFrame(
            data,
            columns=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
        )

        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )


class AmazonAutomotive(DatasetBase):
    r"""AmazonAutomotive.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonAutomotive Class."""
        super().__init__(
            dataset_name="amazon-automotive",
            root_dir=root_dir,
            url=AMAZON_Automotive_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a dataframe consist of the user-item interaction,
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, "amazon-automotive.json.gz")
        print(f"file_name: {file_name}")
        if not os.path.exists(file_name):
            self.download()

        # parse json data
        data = get_data_frame_from_gzip_file(file_name)

        # rename columns
        data = data.rename(
            columns={
                "reviewerID": DEFAULT_USER_COL,
                "asin": DEFAULT_ITEM_COL,
                "overall": DEFAULT_RATING_COL,
                "unixReviewTime": DEFAULT_TIMESTAMP_COL,
            }
        )

        # select necessary columns
        data = pd.DataFrame(
            data,
            columns=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
        )

        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )


class AmazonPetSupplies(DatasetBase):
    r"""AmazonPetSupplies.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonPetSupplies Class."""
        super().__init__(
            dataset_name="amazon-pet-suppplies",
            root_dir=root_dir,
            url=AMAZON_Automotive_URL,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a dataframe consist of the user-item interaction,
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, "amazon-pet-suppplies.json.gz")
        print(f"file_name: {file_name}")
        if not os.path.exists(file_name):
            self.download()

        # parse json data
        data = get_data_frame_from_gzip_file(file_name)

        # rename columns
        data = data.rename(
            columns={
                "reviewerID": DEFAULT_USER_COL,
                "asin": DEFAULT_ITEM_COL,
                "overall": DEFAULT_RATING_COL,
                "unixReviewTime": DEFAULT_TIMESTAMP_COL,
            }
        )

        # select necessary columns
        data = pd.DataFrame(
            data,
            columns=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
        )

        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

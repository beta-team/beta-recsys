import os

import pandas as pd
import gzip

from beta_rec.datasets.dataset_base import DatasetBase
from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
    DEFAULT_TIMESTAMP_COL,
)

# Download URL.
AMAZON_Amazon_Instant_Video_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Amazon_Instant_Video.json.gz"
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
        data = self.get_data_frame_from_gzip_file(file_name)

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

    def parse_gzip_file(self, path):
        g = gzip.open(path, "rb")
        for l in g:
            yield eval(l)

    def get_data_frame_from_gzip_file(self, path):
        i = 0
        df = {}
        for d in self.parse_gzip_file(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient="index")

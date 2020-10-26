import os
import time

import pandas as pd

from ..datasets.dataset_base import DatasetBase
from ..utils.constants import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL

# Download URL
ALIMOBILE_URL = "https://tianchi.aliyun.com/dataset/dataDetail?dataId=46"

# processed data url
ALIMOBILE_RANDOM_SPLIT_URL = "https://1drv.ms/u/s!AjMahLyQeZqughgIvkt5esnpJ3lV?e=bmT3ns"
ALIMOBILE_TEMPORAL_SPLIT_URL = (
    "https://1drv.ms/u/s!AjMahLyQeZqughqYQghbjw_MJqG5?e=9dkaed"
)

# Tips
TIPS = """
AliMobile dataset can not be downloaded by this url automatically, and you need to do:
    1. Download this dataset via 'https://tianchi.aliyun.com/dataset/dataDetail?dataId=46',
    2. Put 'tianchi_mobile_recommend_train_user.zip' into the directory `ali_mobile/raw`,
    3. Unzip 'tianchi_mobile_recommend_train_user.zip',
    4. Rename 'tianchi_mobile_recommend_train_user.csv' to 'ali_mobile.csv'
    5. Rerun this command.
"""


def process_time(standard_time=None):
    r"""Transform time format "xxxx-xx-xxTxx-xx-xxZ" into format "xxxx-xx-xx xx-xx-xx".

    Transform a standard time into our specified format.

    Args:
        standard_time: str with format "xxxx-xx-xxTxx-xx-xxZ".

    Returns:
        timestamp: timestamp data.

    """
    standard_time = standard_time + ":00:00"
    date_arr = time.strptime(standard_time, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(date_arr))
    return timestamp


class AliMobile(DatasetBase):
    r"""AliMobile Dataset.

    AliMobile dataset. This dataset is used to develop an individualized recommendation system of all items,
    it is similar to the taobao dataset.

    The dataset can not be download by the url, you need to down the dataset by
    'https://tianchi.aliyun.com/dataset/dataDetail?dataId=46' and then put it into the directory `ali_mobile/raw`
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init the AliMobile Class."""
        super().__init__(
            "ali_mobile",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            manual_download_url=ALIMOBILE_URL,
            processed_random_split_url=ALIMOBILE_RANDOM_SPLIT_URL,
            processed_temporal_split_url=ALIMOBILE_TEMPORAL_SPLIT_URL,
            tips=TIPS,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory

        Download datasets if not existed.
        ali_mobile_name: UserBehavior.csv

        1. Download ali_mobile dataset if this dataset is not existed.
        2. Load AliMobile <ali-mobile-interaction> table from 'tianchi_mobile_recommend_train_user.csv'.
        3. Save dataset model.
        """
        # Step 1: Download AliMobile dataset if this dataset is not existed.
        ali_mobile_path = os.path.join(self.raw_path, "ali_mobile.csv")
        if not os.path.exists(ali_mobile_path):
            self.download()

        # Step 2: Load AliMobile <ali-mobile-interaction> table from 'ali_mobile.csv.csv'.
        prior_transactions = pd.read_csv(
            ali_mobile_path,
            encoding="utf-8",
            engine="python",
            header=0,
            usecols=[0, 1, 5],
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
        )
        # Add rating column into the dataset.
        prior_transactions.insert(2, "col_rating", 1.0)
        # Transform time data into timestamp format.
        prior_transactions[DEFAULT_TIMESTAMP_COL] = prior_transactions[
            DEFAULT_TIMESTAMP_COL
        ].apply(lambda t: process_time(t))

        # Check the validation of this dataset.
        print(prior_transactions.head())

        # Step 3: Save dataset model.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")

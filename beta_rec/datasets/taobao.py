import os

import pandas as pd

from ..datasets.dataset_base import DatasetBase
from ..utils.constants import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL

# Download URL
TAOBAO_URL = "https://tianchi.aliyun.com/dataset/dataDetail?dataId=649"

# Tips
TAOBAO_TIPS = """
    Taobao dataset can not be downloaded by this url automatically, and you need to do:
    1. Download this dataset via 'https://tianchi.aliyun.com/dataset/dataDetail?dataId=649',
    2. Put 'UserBehavior.csv.zip' into the directory `retailrocket/raw/taobao`,
    3. Unzip 'UserBehavior.csv.zip',
    4. Rename 'UserBehavior.csv' into 'taobao.csv',
    4. Rerun this program.
"""


class Taobao(DatasetBase):
    """Taobao Dataset.

    This dataset is created by randomly selecting about 1 million users who have
    behaviors including click, purchase, adding item to shopping cart and item
    favoring during November 25 to December 03, 2017.

    The dataset is organized in a very similar form to MovieLens-20M, i.e., each
    line represents a specific user-item interaction, which consists of user ID,
    item ID, item's category ID, behavior type and timestamp, separated by commas.

    The dataset can not be download by the url,
    you need to down the dataset by 'https://tianchi.aliyun.com/dataset/dataDetail?dataId=649'
    then put it into the directory `taobao/raw`.
    """

    def __init__(self, dataset_name="taobao", min_u_c=0, min_i_c=3, root_dir=None):
        """Init Taobao Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            manual_download_url=TAOBAO_URL,
            tips=TAOBAO_TIPS,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.

        Download datasets if not existed.
        taobao_name: UserBehavior.csv.

        1. Download taobao dataset if this dataset is not existed.
        2. Load taobao <taobao-interaction> table from 'taobao.csv'.
        3. Save dataset model.
        """
        # Step 1: Download taobao dataset if this dataset is not existed.
        taobao_path = os.path.join(self.raw_path, self.dataset_name, "taobao.csv")
        if not os.path.exists(taobao_path):
            self.download()

        # Step 2: Load taobao <taobao-interaction> table from 'taobao.csv'.
        prior_transactions = pd.read_csv(
            taobao_path,
            encoding="utf-8",
            engine="python",
            header=None,
            usecols=[0, 1, 4],
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
        )
        # Add rating column into the dataset.
        prior_transactions.insert(2, "col_rating", 1.0)

        # Check the validation of this dataset.
        print(prior_transactions.head())

        # Step 3: Save dataset model.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")

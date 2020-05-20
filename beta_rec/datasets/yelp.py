import os
import json
import time
import pandas as pd
from beta_rec.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)
from beta_rec.datasets.dataset_base import DatasetBase

# Download URL.
YELP_URL = "https://www.yelp.com/dataset"

# Yelp
YELP_TIPS = """
    Yelp dataset can not be downloaded by this url automatically, and you need to do:
    1. Download this dataset via 'https://www.yelp.com/dataset',
    2. Put 'yelp-dataset.zip' into the directory `yelp/raw/yelp`,
    3. Unzip 'yelp-dataset.zip',
    4. Rerun this program.
"""


class Yelp(DatasetBase):
    def __init__(self):
        """Yelp

        Yelp dataset.
        The dataset can not be download by the url,
        you need to down the dataset by 'https://www.yelp.com/dataset'
        then put it into the directory `yelp/raw/yelp`
        """
        super().__init__(
            "yelp",
            manual_download_url=YELP_URL,
            processed_leave_one_out_url="",
            processed_random_split_url="",
            processed_temporal_split_url="",
            tips=YELP_TIPS,
        )

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = os.path.join(
            self.raw_path, self.dataset_name, "yelp_academic_dataset_review.json"
        )
        if not os.path.exists(file_name):
            self.download()

        """Load yelp json-format dataset into yelp dataframe.

        1. Load json data in lists, we only use userID, businessID, stars, date in this file.
        2. Add lists into dataframe.
        3. Map fix-length string ID into int format.
        """
        userList, itemList, starList, dateList = [], [], [], []
        userMap, itemMap = {}, {}
        userCnt, itemCnt = 0, 0
        with open(file_name, "r", encoding="utf-8") as fin:
            for line in fin:
                line = json.loads(line)
                user = str(line["user_id"])
                item = str(line["business_id"])
                star = line["stars"]

                # Create timestamp.
                dateStr = str(line["date"])
                dateArr = time.strptime(dateStr, "%Y-%m-%d %H:%M:%S")
                timestamp = int(time.mktime(dateArr))

                # Construct HashMap.
                if user not in userMap:
                    userMap[user] = userCnt
                    userCnt += 1
                if item not in itemMap:
                    itemMap[item] = itemCnt
                    itemCnt += 1

                # Add pairs into dataframe.
                userList.append(user)
                itemList.append(item)
                starList.append(star)
                dateList.append(timestamp)

        prior_transactions = pd.DataFrame(
            {
                DEFAULT_USER_COL: userList,
                DEFAULT_ITEM_COL: itemList,
                DEFAULT_RATING_COL: starList,
                DEFAULT_TIMESTAMP_COL: dateList,
            }
        )

        # Transfer fix-length string into num.
        prior_transactions[DEFAULT_USER_COL] = prior_transactions[
            DEFAULT_USER_COL
        ].apply(lambda u: userMap[u])
        prior_transactions[DEFAULT_ITEM_COL] = prior_transactions[
            DEFAULT_ITEM_COL
        ].apply(lambda i: itemMap[i])

        # Check the validation of this table.
        print(prior_transactions.head())

        # Save this table.
        self.save_dataframe_as_npz(
            prior_transactions,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

        print("Done.")

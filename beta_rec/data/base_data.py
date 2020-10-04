import random

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from ..data.data_loaders import PairwiseNegativeDataset, RatingDataset
from ..utils.alias_table import AliasTable
from ..utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_USER_COL


class BaseData(object):
    r"""A plain DataBase object modeling general recommendation data. Re_index all the users and items from raw dataset.

    Args:
        split_dataset (train,valid,test): the split dataset, a tuple consisting of training (DataFrame),
            validate/list of validate (DataFrame), testing/list of testing (DataFrame).
        intersect (bool, optional): remove users and items of test/valid sets that do not exist in the train set. If the
            model is able to predict for new users and new items, this can be :obj:`False`. (default: :obj:`True`).
        binarize (bool, optional): binarize the rating column of train set 0 or 1, i.e. implicit feedback.
            (default: :obj:`True`).
        bin_thld (int, optional): the threshold of binarization (default: :obj:`0`) normalize (bool, optional): normalize
            the rating column of train.
            set into [0, 1], i.e. explicit feedback. (default: :obj:`False`).
    """

    def __init__(
        self,
        split_dataset,
        intersect=True,
        binarize=True,
        bin_thld=0.0,
        normalize=False,
    ):
        """Initialize BaseData Class."""
        self.train, self.valid, self.test = split_dataset
        self.user_pool = list(self.train[DEFAULT_USER_COL].unique())
        self.item_pool = list(self.train[DEFAULT_ITEM_COL].unique())
        self.n_users = len(self.user_pool)
        self.n_items = len(self.item_pool)
        self.user_id_pool = [i for i in range(self.n_users)]
        self.item_id_pool = [i for i in range(self.n_items)]

        if intersect:
            self._intersect()

        if binarize:
            self._binarize(bin_thld)

        if normalize:
            self._normalize()

        self._re_index()

        self.item_sampler = AliasTable(
            self.train[DEFAULT_ITEM_COL].value_counts().to_dict()
        )
        self.user_sampler = AliasTable(
            self.train[DEFAULT_USER_COL].value_counts().to_dict()
        )

    def _binarize(self, bin_thld):
        """Binarize ratings into 0 or 1, i.e. implicit feedback."""
        for data in [self.train, self.valid, self.test]:
            if isinstance(data, list):
                for sub_data in data:
                    sub_data.loc[
                        sub_data[DEFAULT_RATING_COL] > bin_thld, DEFAULT_RATING_COL
                    ] = 1.0
            else:
                data.loc[data[DEFAULT_RATING_COL] > bin_thld, DEFAULT_RATING_COL] = 1.0

    def _normalize(self):
        """Normalize ratings into [0, 1] from [0, max_rating], explicit feedback."""
        max_rating = self.train[DEFAULT_RATING_COL].max()
        assert max_rating > 0, "All rating may be less than 0 (or not be a number)."

        for data in [self.train, self.valid, self.test]:
            if isinstance(data, list):
                for sub_data in data:
                    sub_data.loc[
                        :, DEFAULT_RATING_COL
                    ] = sub_data.DEFAULT_RATING_COL.apply(
                        lambda x: x * 1.0 / max_rating
                    )
            else:
                data.loc[:, DEFAULT_RATING_COL] = data.DEFAULT_RATING_COL.apply(
                    lambda x: x * 1.0 / max_rating
                )

    def _re_index(self):
        """Reindex for list of dataset.

        For example, validate and test can be a list for evaluation.
        """
        # Reindex user and item index
        self.user2id = dict(zip(np.array(self.user_pool), np.arange(self.n_users)))
        self.id2user = {self.user2id[k]: k for k in self.user2id}

        self.item2id = dict(zip(np.array(self.item_pool), np.arange(self.n_items)))
        self.id2item = {self.item2id[k]: k for k in self.item2id}

        for data in [self.train, self.valid, self.test]:
            if isinstance(data, list):
                for sub_data in data:
                    # Map user_idx and item_idx
                    sub_data.loc[:, DEFAULT_USER_COL] = sub_data[
                        DEFAULT_USER_COL
                    ].apply(lambda x: self.user2id[x])
                    sub_data.loc[:, DEFAULT_ITEM_COL] = sub_data[
                        DEFAULT_ITEM_COL
                    ].apply(lambda x: self.item2id[x])
            else:
                # Map user_idx and item_idx
                data.loc[:, DEFAULT_USER_COL] = data[DEFAULT_USER_COL].apply(
                    lambda x: self.user2id[x]
                )
                data.loc[:, DEFAULT_ITEM_COL] = data[DEFAULT_ITEM_COL].apply(
                    lambda x: self.item2id[x]
                )

    def _intersect(self):
        """Intersect validation and test datasets with train datasets and then reindex userID and itemID."""
        for data in [self.valid, self.test]:
            if isinstance(data, list):
                for sub_data in data:
                    sub_data.drop(
                        sub_data[
                            ~sub_data[DEFAULT_USER_COL].isin(self.user_pool)
                        ].index,
                        inplace=True,
                    )
                    sub_data.drop(
                        sub_data[
                            ~sub_data[DEFAULT_ITEM_COL].isin(self.item_pool)
                        ].index,
                        inplace=True,
                    )
            else:
                data.drop(
                    data[~data[DEFAULT_USER_COL].isin(self.user_pool)].index,
                    inplace=True,
                )
                data.drop(
                    data[~data[DEFAULT_ITEM_COL].isin(self.item_pool)].index,
                    inplace=True,
                )

        print("After intersection, testing set [0] statistics")
        print(
            tabulate(
                self.test[0].agg(["count", "nunique"])
                if isinstance(self.test, list)
                else self.test.agg(["count", "nunique"]),
                headers=self.test[0].columns
                if isinstance(self.test, list)
                else self.test.columns,
                tablefmt="psql",
                disable_numparse=True,
            )
        )
        print("After intersection, validation set [0] statistics")
        print(
            tabulate(
                self.valid[0].agg(["count", "nunique"])
                if isinstance(self.test, list)
                else self.test.agg(["count", "nunique"]),
                headers=self.test[0].columns
                if isinstance(self.test, list)
                else self.test.columns,
                tablefmt="psql",
                disable_numparse=True,
            )
        )

    def instance_bce_loader(self, batch_size, device, num_negative):
        """Instance a train DataLoader that have rating."""
        users, items, ratings = [], [], []
        interact_status = (
            self.train.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL]
            .apply(set)
            .reset_index()
            .rename(columns={DEFAULT_ITEM_COL: "positive_items"})
        )
        interact_status["negative_items"] = interact_status["positive_items"].apply(
            lambda x: set(self.item_id_pool) - x
        )
        train_ratings = pd.merge(
            self.train,
            interact_status[[DEFAULT_USER_COL, "negative_items"]],
            on=DEFAULT_USER_COL,
        )
        train_ratings["negative_samples"] = train_ratings["negative_items"].apply(
            lambda x: random.sample(list(x), num_negative)
        )
        for _, row in train_ratings.iterrows():
            users.append(int(row[DEFAULT_USER_COL]))
            items.append(int(row[DEFAULT_ITEM_COL]))
            ratings.append(float(row[DEFAULT_RATING_COL]))
            for i in range(num_negative):
                users.append(int(row[DEFAULT_USER_COL]))
                items.append(int(row["negative_samples"][i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = RatingDataset(
            user_tensor=torch.LongTensor(users).to(device),
            item_tensor=torch.LongTensor(items).to(device),
            target_tensor=torch.FloatTensor(ratings).to(device),
        )
        print(f"Making RatingDataset of length {len(dataset)}")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def instance_bpr_loader(self, batch_size, device):
        """Instance a pairwise Data_loader for training.

        Sample ONE negative items for each user-item pare, and shuffle them with positive items.
        A batch of data in this DataLoader is suitable for a binary cross-entropy loss.
        # todo implement the item popularity-biased sampling
        """
        users, pos_items, neg_items = [], [], []
        interact_status = (
            self.train.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL]
            .apply(set)
            .reset_index()
            .rename(columns={DEFAULT_ITEM_COL: "positive_items"})
        )
        interact_status["negative_items"] = interact_status["positive_items"].apply(
            lambda x: set(self.item_id_pool) - x
        )
        train_ratings = pd.merge(
            self.train,
            interact_status[[DEFAULT_USER_COL, "negative_items"]],
            on=DEFAULT_USER_COL,
        )
        train_ratings["negative_sample"] = train_ratings["negative_items"].apply(
            lambda x: random.sample(list(x), 1)[0]
        )
        for _, row in train_ratings.iterrows():
            users.append(row[DEFAULT_USER_COL])
            pos_items.append(row[DEFAULT_ITEM_COL])
            neg_items.append(row["negative_sample"])
        dataset = PairwiseNegativeDataset(
            user_tensor=torch.LongTensor(users).to(device),
            pos_item_tensor=torch.LongTensor(pos_items).to(device),
            neg_item_tensor=torch.LongTensor(neg_items).to(device),
        )
        print(f"Making PairwiseNegativeDataset of length {len(dataset)}")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from beta_rec.utils.alias_table import AliasTable
from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)


class DataBase():
    def __init__(self, split_dataset, intersection=True, binarize=True, ):
        self.train, self.valid, self.test = split_dataset
        
        self.user_pool = set(self.train[DEFAULT_USER_COL].unique())
        self.item_pool = set(self.train[DEFAULT_ITEM_COL].unique())
        self.n_users = len(self.user_pool)
        self.n_items = len(self.item_pool)
        
        if intersection:
            self._intersection()
        if binarize:
            self._binarize()

        self.item_sampler = AliasTable(
            self.train[DEFAULT_ITEM_COL].value_counts().to_dict()
        )
        self.user_sampler = AliasTable(
            self.train[DEFAULT_USER_COL].value_counts().to_dict()
        )
        
    def _binarize(self):
        """Binarize ratings into 0 or 1, imlicit feedback."""
        for data in [self.train, self.valid, self.test]:
            if isinstance(data, list):
                for sub_data in data:
                    sub_data[DEFAULT_RATING_COL][data[DEFAULT_RATING_COL] > 0] = 1.0
            else:
                data[DEFAULT_RATING_COL][data[DEFAULT_RATING_COL] > 0] = 1.0
        
    def _normalize(self):
        """Normalize ratings into [0, 1] from [0, max_rating], explicit feedback."""
        ratings = deepcopy(ratings)
        max_rating = self.train[DEFAULT_RATING_COL].max()
        for data in [self.train, self.valid, self.test]:
            if isinstance(data, list):
                for sub_data in data:
                    sub_data[DEFAULT_RATING_COL][data[DEFAULT_RATING_COL] > 0] = 1.0
            else:    
                data[DEFAULT_RATING_COL] = data[DEFAULT_RATING_COL] * 1.0 / max_rating
    
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
                    sub_data = sub_data[data[DEFAULT_USER_COL].isin(self.user_pool)
                                        & data[DEFAULT_ITEM_COL].isin(self.item_pool)
                                       ]
                    # Map user_idx and item_idx
                    sub_data[DEFAULT_USER_COL] = sub_data[DEFAULT_USER_COL].apply(lambda x: self.user2id[x])
                    sub_data[DEFAULT_ITEM_COL] = sub_data[DEFAULT_ITEM_COL].apply(lambda x: self.item2id[x])
            else:
                data = data[
                    data[DEFAULT_USER_COL].isin(self.user_pool)
                    & data[DEFAULT_ITEM_COL].isin(self.item_pool)
                ]
                # Map user_idx and item_idx
                data[DEFAULT_USER_COL] = data[DEFAULT_USER_COL].apply(lambda x: self.user2id[x])
                data[DEFAULT_ITEM_COL] = data[DEFAULT_ITEM_COL].apply(lambda x: self.item2id[x])
                
        
    def _intersection(self):
        """Intersect validation and test datasets with train datasets and then reindex userID and itemID

        """
        for data in [self.valid, self.test]:
                if isinstance(data, list):
                    for sub_data in data:
                         sub_data = sub_data[sub_data[DEFAULT_USER_COL].isin(user_pool)
                                             & sub_data[DEFAULT_ITEM_COL].isin(items_pool)
                                            ]
                else:
                    data = data[data[DEFAULT_USER_COL].isin(user_pool)
                                & data[DEFAULT_ITEM_COL].isin(items_pool)
                               ]
                
        print("After intersection, test statistics")
        print(
            tabulate(
                self.test[0].agg(["count", "nunique"]) if isinstance(self.test, list) 
                else self.test.agg(["count", "nunique"]),
                headers=self.test[0].columns if isinstance(self.test, list) else self.test.columns,
                tablefmt="psql",
                disable_numparse=True,
            )
        )
        
        self._re_index()
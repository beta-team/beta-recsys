import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_ORDER_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)


class Sampler(object):
    """Sampler Class."""

    def __init__(self, df_train, sample_file, n_sample, dump=True, load_save=False):
        """Init Sampler Class."""
        self.sample_file = sample_file
        self.df_train = df_train
        self.n_sample = n_sample
        self.dump = dump
        self.load_save = load_save
        print("Initialize Sampler!")

    def sample(self):
        """Generate samples."""
        if self.load_save:
            if os.path.exists(self.sample_file):
                return self.load_triples_from_file(self.sample_file)
        print("Preparing training triples ... ")
        self.dataTrain = (
            self.df_train.groupby([DEFAULT_ORDER_COL, DEFAULT_USER_COL])[
                DEFAULT_ITEM_COL
            ]
            .apply(list)
            .reset_index()
        )
        self.dataTrain.rename(
            columns={
                DEFAULT_USER_COL: "UID",
                DEFAULT_ORDER_COL: "TID",
                DEFAULT_ITEM_COL: "PID",
            },
            inplace=True,
        )
        n_orders = self.dataTrain.shape[0]
        sampled_index = np.random.choice(n_orders, size=self.n_sample)
        sampled_order = self.dataTrain.iloc[sampled_index].reset_index()

        process_bar = tqdm(
            range(self.n_sample),
            file=sys.stdout,
            miniters=int(self.n_sample / 20),
            maxinterval=60,
        )
        res = []
        for i in process_bar:
            _index, _tid, _uid, _items = sampled_order.iloc[i]
            _i, _j = np.random.choice(_items, size=2)
            res.append([int(_uid), int(_i), int(_j)])
        print("done!")
        data_dic = {}
        res = np.array(res)
        data_dic["UID"] = res[:, 0]
        data_dic["PID1"] = res[:, 1]
        data_dic["PID2"] = res[:, 2]
        triple_df = pd.DataFrame(data_dic)
        if self.dump:
            triple_df.to_csv(self.sample_file, index=False)
        return triple_df

    def sample_by_time(self, time_step):
        """Generate samples by time."""
        if self.load_save:
            if os.path.exists(self.sample_file):
                return self.load_triples_from_file(self.sample_file)
        if time_step == 0:
            return self.sample()
        print("preparing training triples ... ")
        self.dataTrain = (
            self.df_train.groupby([DEFAULT_ORDER_COL, DEFAULT_USER_COL])[
                DEFAULT_ITEM_COL
            ]
            .apply(list)
            .reset_index()
        )
        dataTrain_timestep = (
            self.df_train.groupby([DEFAULT_ORDER_COL])[DEFAULT_TIMESTAMP_COL]
            .apply(lambda a: a.mean())
            .reset_index()
        )
        self.dataTrain = self.dataTrain.merge(dataTrain_timestep)
        self.dataTrain = self.dataTrain.sort_values(by=DEFAULT_TIMESTAMP_COL)
        self.dataTrain.rename(
            columns={
                DEFAULT_USER_COL: "UID",
                DEFAULT_ORDER_COL: "TID",
                DEFAULT_ITEM_COL: "PID",
            },
            inplace=True,
        )
        n_orders = self.dataTrain.shape[0]
        n_orders_per_t = int(n_orders / time_step)
        n_sample_per_t = int(self.n_sample / time_step)
        process_bar = tqdm(range(time_step), file=sys.stdout, maxinterval=60)
        rest_baskets = n_orders - time_step * n_orders_per_t
        res = []
        for t in process_bar:
            if t != 0:
                index_start = t * n_orders_per_t + rest_baskets
                index_end = (t + 1) * n_orders_per_t + rest_baskets
            else:
                index_start = 0
                index_end = rest_baskets
            sampled_index = np.random.choice(
                np.arange(index_start, index_end),
                size=n_sample_per_t,
            )
            sampled_order = self.dataTrain.iloc[sampled_index]
            for _, row in sampled_order.iterrows():
                _uid, _, _items = row["UID"], row["TID"], row["PID"]
                _i, _j = np.random.choice(_items, size=2)
                res.append([int(_uid), int(_i), int(_j), int(t)])
        res = np.array(res)
        data_dic = {}
        data_dic["UID"] = res[:, 0]
        data_dic["PID1"] = res[:, 1]
        data_dic["PID2"] = res[:, 2]
        data_dic["T"] = res[:, 3]
        triple_df = pd.DataFrame(data_dic)
        if self.dump:
            triple_df.to_csv(self.sample_file, index=False)
        return triple_df

    def load_triples_from_file(self, triple_file):
        """Load triples from file."""
        print("Loading triples from file:", triple_file)
        return pd.read_csv(triple_file)

import os
import random
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from beta_rec.data.auxiliary_data import Auxiliary
from beta_rec.data.base_data import BaseData
from beta_rec.datasets.data_load import load_user_item_feature
from beta_rec.utils.common_util import ensureDir, normalized_adj_single
from beta_rec.utils.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from beta_rec.utils.triple_sampler import Sampler

pd.options.mode.chained_assignment = None  # default='warn'


class GroceryData(BaseData, Auxiliary):
    r"""A Grocery Data object, which consist one more order/basket column than the BaseData. Re_index all the users and items from raw dataset.

        Args:
            split_dataset (train,valid,test): the split dataset, a tuple consisting of training (DataFrame),
            validate/list of validate (DataFrame), testing/list of testing (DataFrame).
            intersect (bool, optional): remove users and items of test/valid sets that do not exist in the train set.
            If the model is able to predict for new users and new items, this can be :obj:`False`.
            (default: :obj:`True`)
            binarize (bool, optional): binarize the rating column of train set 0 or 1, i.e. implicit feedback.
            (default: :obj:`True`)
            bin_thld (int, optional):  the threshold of binarization (default: :obj:`0`)
            normalize (bool, optional): normalize the rating column of train set into [0, 1], i.e. explicit feedback.
            (default: :obj:`False`)

    """

    def __init__(
        self,
        split_dataset,
        config,
        intersect=True,
        binarize=True,
        bin_thld=0.0,
        normalize=False,
    ):
        BaseData.__init__(
            split_dataset,
            intersect=intersect,
            binarize=binarize,
            bin_thld=bin_thld,
            normalize=normalize,
        )
        self.config = config
        Auxiliary.__init__(config, self.n_users, self.n_items)

    def sample_triple_time(self, dump=True, load_save=False):
        """Sample triples or load triples samples from files.

        This method is only applicable for basket based Recommender.

        Returns:
            None

        """
        sample_file_name = (
            "triple_"
            + self.config["dataset"]["dataset"]
            + (
                ("_" + str(self.config["dataset"]["percent"] * 100))
                if "percent" in self.config
                else ""
            )
            + (
                ("_" + str(self.config["model"]["time_step"]))
                if "time_step" in self.config
                else "_10"
            )
            + "_"
            + str(self.config["model"]["n_sample"])
            if "percent" in self.config
            else "" + ".csv"
        )
        self.process_path = self.config["system"]["process_dir"]
        ensureDir(self.process_path)
        sample_file = os.path.join(self.process_path, sample_file_name)
        my_sampler = Sampler(
            self.train,
            sample_file,
            self.config["model"]["n_sample"],
            dump=dump,
            load_save=load_save,
        )
        return my_sampler.sample_by_time(self.config["model"]["time_step"])

    def sample_triple(self, dump=True, load_save=False):
        """Sample triples or load triples samples from files.

        This method is only applicable for basket based Recommender.

        Returns:
            None

        """
        sample_file_name = (
            "triple_"
            + self.config["dataset"]["dataset"]
            + (
                ("_" + str(self.config["dataset"]["percent"] * 100))
                if "percent" in self.config
                else ""
            )
            + "_"
            + str(self.config["model"]["n_sample"])
            if "percent" in self.config
            else "" + ".csv"
        )
        self.process_path = self.config["system"]["process_dir"]
        ensureDir(self.process_path)
        sample_file = os.path.join(self.process_path, sample_file_name)
        my_sampler = Sampler(
            self.train,
            sample_file,
            self.config["model"]["n_sample"],
            dump=dump,
            load_save=load_save,
        )
        return my_sampler.sample()

    def get_adj_mat(self):
        """Get the adjacent matrix.

        If not previously stored then call the function to create. This method is for NGCF model.

        Return:
            Different types of adjacent matrix.
        """
        self.init_train_items()

        process_file_name = (
            "ngcf_"
            + self.config["dataset"]["dataset"]
            + "_"
            + self.config["dataset"]["data_split"]
            + (
                ("_" + str(self.config["dataset"]["percent"] * 100))
                if "percent" in self.config
                else ""
            )
        )
        self.process_path = os.path.join(
            self.config["system"]["process_dir"],
            self.config["dataset"]["dataset"] + "/",
        )
        process_file_name = os.path.join(self.process_path, process_file_name)
        ensureDir(process_file_name)
        print(process_file_name)
        try:
            t1 = time()
            adj_mat = sp.load_npz(os.path.join(process_file_name, "s_adj_mat.npz"))
            norm_adj_mat = sp.load_npz(
                os.path.join(process_file_name, "s_norm_adj_mat.npz")
            )
            mean_adj_mat = sp.load_npz(
                os.path.join(process_file_name, "s_mean_adj_mat.npz")
            )
            print("already load adj matrix", adj_mat.shape, time() - t1)
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(os.path.join(process_file_name, "s_adj_mat.npz"), adj_mat)
            sp.save_npz(
                os.path.join(process_file_name, "s_norm_adj_mat.npz"), norm_adj_mat
            )
            sp.save_npz(
                os.path.join(process_file_name, "s_mean_adj_mat.npz"), mean_adj_mat
            )
        return adj_mat, norm_adj_mat, mean_adj_mat

    def load_user_item_fea(self):
        """Load user and item features from datasets.

        Returns:

        """
        print("Load user and item features from datasets")
        user_feat, item_feat = load_user_item_feature(self.config)
        user_feat_li = [None for i in range(self.n_users)]
        item_feat_li = [None for i in range(self.n_items)]
        for user in user_feat:
            if user[0] in self.user2id:
                user_feat_li[self.user2id[user[0]]] = user[1:]
        for item in item_feat:
            if item[0] in self.item2id:
                item_feat_li[self.item2id[item[0]]] = item[1:]
        self.user_feat = np.stack(user_feat_li)
        self.item_feat = np.stack(item_feat_li)

    def create_adj_mat(self):
        """Create adjacent matrix from the user-item interaction."""
        t1 = time()
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        adj_mat[: self.n_users, self.n_users :] = R
        adj_mat[self.n_users :, : self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print("already create adjacency matrix", adj_mat.shape, time() - t1)
        t2 = time()
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        print("already normalize adjacency matrix", time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [random.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print("refresh negative pools", time() - t1)

    def sample(self, batch_size):
        """Sample users, their positive items and negative items.

        Returns:
            users (list)
            pos_items (list)
            neg_items (list)
        """
        if batch_size <= self.n_users:
            users = random.sample(range(self.n_users), batch_size)
        else:
            users = [random.choice(range(self.n_users)) for _ in range(batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return random.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        return users, pos_items, neg_items

    def sample_all_users_pos_items(self):
        self.all_train_users = []

        self.all_train_pos_items = []
        for u in range(self.n_users):
            self.all_train_users += [u] * len(self.train_items[u])
            self.all_train_pos_items += self.train_items[u]

    def epoch_sample(self):
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        neg_items = []
        for u in self.all_train_users:
            neg_items += sample_neg_items_for_u(u, 1)

        perm = np.random.permutation(len(self.all_train_users))
        users = np.array(self.all_train_users)[perm]
        pos_items = np.array(self.all_train_pos_items)[perm]
        neg_items = np.array(neg_items)[perm]
        return users, pos_items, neg_items

    def init_train_items(self):
        self.n_train = 0
        self.train_items = {}
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        user_np = np.array(self.train[DEFAULT_USER_COL])
        item_np = np.array(self.train[DEFAULT_ITEM_COL])
        for u in range(self.n_users):
            index = list(np.where(user_np == u)[0])
            i = item_np[index]
            self.train_items[u] = i
            for item in i:
                self.R[u, item] = 1
                self.n_train += 1

    def neighbour_process(self):
        user_np = np.array(self.train[DEFAULT_USER_COL])
        item_np = np.array(self.train[DEFAULT_ITEM_COL])
        # Neighborhods
        users_items = defaultdict(set)
        items_users = defaultdict(set)
        zip_list = list(zip(user_np, item_np))
        for u, i in zip_list:
            users_items[u].add(i)
            items_users[i].add(u)
        train_data = np.array(zip_list)
        train_index = np.arange(len(train_data), dtype=np.uint)
        # get the list version
        item_users_list = {k: list(v) for k, v in items_users.items()}
        max_user_neighbors = max([len(x) for x in items_users.values()])
        users_items = dict(users_items)
        items_users = dict(items_users)

        return (
            train_data,
            train_index,
            max_user_neighbors,
            items_users,
            users_items,
            item_users_list,
        )

    def cmn_train_loader(self, batch_size: int, neighborhood: bool, neg_count: int):
        """
        Batch data together as (user, item, negative item), pos_neighborhood,
        length of neighborhood, negative_neighborhood, length of negative neighborhood

        if neighborhood is False returns only user, item, negative_item so we
        can reuse this for non-neighborhood-based methods.

        :param batch_size: size of the batch
        :param neighborhood: return the neighborhood information or not
        :param neg_count: number of negative samples to uniformly draw per a pos
                          example
        :return: generator
        """
        # Allocate inputs
        (
            train_data,
            train_index,
            max_user_neighbors,
            items_users,
            users_items,
            item_users_list,
        ) = self.neighbour_process()
        self.item_users_list = item_users_list
        batch = np.zeros((batch_size, 3), dtype=np.uint32)
        pos_neighbor = np.zeros((batch_size, max_user_neighbors), dtype=np.int32)
        pos_length = np.zeros(batch_size, dtype=np.int32)
        neg_neighbor = np.zeros((batch_size, max_user_neighbors), dtype=np.int32)
        neg_length = np.zeros(batch_size, dtype=np.int32)

        def sample_negative_item(user_id, n_items, users_items, items_users):
            """Uniformly sample a negative item."""
            if user_id > n_items:
                raise ValueError(
                    "Trying to sample user id: {} > user count: {}".format(
                        user_id, n_items
                    )
                )

            n = np.random.randint(0, n_items)
            positive_items = users_items[user_id]

            if len(positive_items) >= n_items:
                raise ValueError(
                    "The User has rated more items than possible %s / %s"
                    % (len(positive_items), n_items)
                )
            while n in positive_items or n not in items_users:
                n = np.random.randint(0, n_items)
            return n

        # Shuffle index
        np.random.shuffle(train_index)

        idx = 0
        for user_idx, item_idx in train_data[train_index]:
            for _ in range(neg_count):
                neg_item_idx = sample_negative_item(
                    user_idx, self.n_items, users_items, items_users
                )
                batch[idx, :] = [user_idx, item_idx, neg_item_idx]

                # Get neighborhood information
                if neighborhood:
                    if len(items_users.get(item_idx, [])) > 0:
                        pos_length[idx] = len(items_users[item_idx])
                        pos_neighbor[idx, : pos_length[idx]] = item_users_list[item_idx]
                    else:
                        # Length defaults to 1
                        pos_length[idx] = 1
                        pos_neighbor[idx, 0] = item_idx

                    if len(items_users.get(neg_item_idx, [])) > 0:
                        neg_length[idx] = len(items_users[neg_item_idx])
                        neg_neighbor[idx, : neg_length[idx]] = item_users_list[
                            neg_item_idx
                        ]
                    else:
                        # Length defaults to 1
                        neg_length[idx] = 1
                        neg_neighbor[idx, 0] = neg_item_idx

                idx += 1
                # Yield batch if we filled queue
                if idx == batch_size:
                    if neighborhood:
                        max_length = max(neg_length.max(), pos_length.max())
                        yield batch, pos_neighbor[
                            :, :max_length
                        ], pos_length, neg_neighbor[:, :max_length], neg_length
                        pos_length[:] = 1
                        neg_length[:] = 1
                    else:
                        yield batch
                    # Reset
                    idx = 0

        # Provide remainder
        if idx > 0:
            if neighborhood:
                max_length = max(neg_length[:idx].max(), pos_length[:idx].max())
                yield batch[:idx], pos_neighbor[:idx, :max_length], pos_length[
                    :idx
                ], neg_neighbor[:idx, :max_length], neg_length[:idx]
            else:
                yield batch[:idx]

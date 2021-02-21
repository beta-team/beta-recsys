import json
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader, Dataset

from ..utils.common_util import ensureDir, normalized_adj_single
from ..utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset."""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """Init UserItemRatingDataset Class.

        Args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair.
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        """Get an item from dataset."""
        return (
            self.user_tensor[index],
            self.item_tensor[index],
            self.target_tensor[index],
        )

    def __len__(self):
        """Get the size of the dataset."""
        return self.user_tensor.size(0)


class RatingNegativeDataset(Dataset):
    """RatingNegativeDataset.

    Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset, which contains negative items with rating
    being 0.0.
    """

    def __init__(self, user_tensor, item_tensor, rating_tensor):
        """Init RatingNegativeDataset Class.

        Args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair.
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.rating_tensor = rating_tensor

    def __getitem__(self, index):
        """Get an item from the dataset.

        Args:
            index:

        Returns: users, pos_items, neg_items, pos_ratings, neg_ratings
        """
        return (
            self.user_tensor[index],
            self.item_tensor[index],
            self.rating_tensor[index],
        )

        # index = torch.LongTensor(index, device=self.user_tensor.device)
        # pos_index = index[self.rating_tensor[index] > 0]
        # neg_index = index[self.rating_tensor[index] >= 0]
        # return (
        #     self.user_tensor[pos_index],
        #     self.item_tensor[pos_index],
        #     self.rating_tensor[pos_index],
        #     self.user_tensor[neg_index],
        #     self.item_tensor[neg_index],
        #     self.rating_tensor[neg_index],
        # )

    def __len__(self):
        """Get the size of the dataset."""
        return self.user_tensor.size(0)


class PairwiseNegativeDataset(Dataset):
    """Wrapper, convert <user, pos_item, neg_item> Tensor into Pytorch Dataset."""

    def __init__(self, user_tensor, pos_item_tensor, neg_item_tensor):
        """Init PairwiseNegativeDataset Class.

        Args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair.
        """
        self.user_tensor = user_tensor
        self.pos_item_tensor = pos_item_tensor
        self.neg_item_tensor = neg_item_tensor

    def __getitem__(self, index):
        """Get an item from the dataset."""
        return (
            self.user_tensor[index],
            self.pos_item_tensor[index],
            self.neg_item_tensor[index],
        )

    def __len__(self):
        """Get the size of the dataset."""
        return self.user_tensor.size(0)


class DataLoaderBase(object):
    """Construct dataset for NCF."""

    def __init__(self, ratings):
        """Init DataLoaderBase Class.

        Args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert DEFAULT_USER_COL in ratings.columns
        assert DEFAULT_ITEM_COL in ratings.columns
        assert DEFAULT_RATING_COL in ratings.columns
        assert DEFAULT_TIMESTAMP_COL in ratings.columns

        self.ratings = ratings
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings[DEFAULT_USER_COL].unique())
        self.item_pool = set(self.ratings[DEFAULT_ITEM_COL].unique())
        self.n_users = len(self.user_pool)
        self.n_items = len(self.item_pool)
        # create negative item samples for NCF learning
        self.negatives = self._sample_negative(ratings)

    def _normalize(self, ratings):
        """Normalize into [0, 1] from [0, max_rating], explicit feedback."""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings[DEFAULT_RATING_COL] = ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        """Binarize into 0 or 1, imlicit feedback."""
        ratings = deepcopy(ratings)
        ratings[DEFAULT_RATING_COL][ratings[DEFAULT_RATING_COL] > 0] = 1.0
        return ratings

    def _sample_negative(self, ratings):
        """Return all negative items & 100 sampled negative items."""
        interact_status = (
            ratings.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL]
            .apply(set)
            .reset_index()
            .rename(columns={DEFAULT_ITEM_COL: "interacted_items"})
        )
        interact_status["negative_items"] = interact_status["interacted_items"].apply(
            lambda x: self.item_pool - x
        )
        interact_status["negative_samples"] = interact_status["negative_items"].apply(
            lambda x: random.sample(x, 99)
        )
        return interact_status[[DEFAULT_USER_COL, "negative_items", "negative_samples"]]

    def instance_a_train_loader(self, num_negatives, batch_size):
        """Instance train loader for one training epoch."""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(
            self.ratings,
            self.negatives[[DEFAULT_USER_COL, "negative_items"]],
            on=DEFAULT_USER_COL,
        )
        train_ratings["negatives"] = train_ratings["negative_items"].apply(
            lambda x: random.sample(x, num_negatives)
        )
        for _, row in train_ratings.iterrows():
            users.append(int(row[DEFAULT_USER_COL]))
            items.append(int(row[DEFAULT_ITEM_COL]))
            ratings.append(float(row[DEFAULT_RATING_COL]))
            for i in range(num_negatives):
                users.append(int(row[DEFAULT_USER_COL]))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(
            user_tensor=torch.LongTensor(users),
            item_tensor=torch.LongTensor(items),
            target_tensor=torch.FloatTensor(ratings),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def uniform_negative_train_loader(self, num_negatives, batch_size, device):
        """Instance a Data_loader for training.

        Sample 'num_negatives' negative items for each user, and shuffle them with positive items.
        A batch of data in this DataLoader is suitable for a binary cross-entropy loss.
        # todo implement the item popularity-biased sampling
        """
        users, items, ratings = [], [], []
        train_ratings = pd.merge(
            self.ratings,
            self.negatives[[DEFAULT_USER_COL, "negative_items"]],
            on=DEFAULT_USER_COL,
        )
        train_ratings["negatives"] = train_ratings["negative_items"].apply(
            lambda x: random.sample(x, num_negatives)
        )
        for _, row in train_ratings.iterrows():
            users.append(int(row[DEFAULT_USER_COL]))
            items.append(int(row[DEFAULT_ITEM_COL]))
            ratings.append(float(row[DEFAULT_RATING_COL]))
            for i in range(num_negatives):
                users.append(int(row[DEFAULT_USER_COL]))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = RatingNegativeDataset(
            user_tensor=torch.LongTensor(users).to(device),
            item_tensor=torch.LongTensor(items).to(device),
            rating_tensor=torch.FloatTensor(ratings).to(device),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def pairwise_negative_train_loader(self, batch_size, device):
        """Instance a pairwise Data_loader for training.

        Sample ONE negative items for each user-item pare, and shuffle them with positive items.
        A batch of data in this DataLoader is suitable for a binary cross-entropy loss.
        # todo implement the item popularity-biased sampling
        """
        users, pos_items, neg_items = [], [], []
        train_ratings = pd.merge(
            self.ratings,
            self.negatives[[DEFAULT_USER_COL, "negative_items"]],
            on=DEFAULT_USER_COL,
        )
        train_ratings["one_negative"] = train_ratings["negative_items"].apply(
            lambda x: random.sample(x, 1)
        )
        for _, row in train_ratings.iterrows():
            users.append(int(row[DEFAULT_USER_COL]))
            pos_items.append(int(row[DEFAULT_ITEM_COL]))
            neg_items.append(int(row.one_negative[0]))
        dataset = PairwiseNegativeDataset(
            user_tensor=torch.LongTensor(users).to(device),
            pos_item_tensor=torch.LongTensor(pos_items).to(device),
            neg_item_tensor=torch.LongTensor(neg_items).to(device),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """Create evaluation data."""
        test_ratings = pd.merge(
            self.test_ratings,
            self.negatives[[DEFAULT_USER_COL, "negative_samples"]],
            on=DEFAULT_USER_COL,
        )
        test_users, test_items, ratings = [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row[DEFAULT_USER_COL]))
            test_items.append(int(row[DEFAULT_ITEM_COL]))
            ratings.append(1)
            for i in range(len(row.negative_samples)):
                test_users.append(int(row[DEFAULT_USER_COL]))
                test_items.append(int(row.negative_samples[i]))
                ratings.append(0)

        test_df = pd.DataFrame(
            {
                DEFAULT_USER_COL: test_users,
                DEFAULT_ITEM_COL: test_items,
                DEFAULT_RATING_COL: ratings,
            }
        )
        return test_df

    def get_adj_mat(self, config):
        """Get the adjacent matrix, if not previously stored then call the function to create.

        This method is for NGCF model.

        Returns:
            Different types of adjacment matrix.
        """
        process_file_name = (
            "ngcf_"
            + config["dataset"]["dataset"]
            + "_"
            + config["dataset"]["data_split"]
            + (
                ("_" + str(config["dataset"]["percent"] * 100))
                if "percent" in config
                else ""
            )
        )
        process_path = os.path.join(
            config["system"]["process_dir"],
            config["dataset"]["dataset"] + "/",
        )
        process_file_name = os.path.join(process_path, process_file_name)
        ensureDir(process_file_name)
        print(process_file_name)
        try:
            adj_mat = sp.load_npz(os.path.join(process_file_name, "s_adj_mat.npz"))
            norm_adj_mat = sp.load_npz(
                os.path.join(process_file_name, "s_norm_adj_mat.npz")
            )
            mean_adj_mat = sp.load_npz(
                os.path.join(process_file_name, "s_mean_adj_mat.npz")
            )
            print("already load adj matrix", adj_mat.shape)
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

    def create_adj_mat(self):
        """Create adjacent matirx from the user-item interaction matrix."""
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()

        R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        user_np = np.array(self.ratings[DEFAULT_USER_COL])
        item_np = np.array(self.ratings[DEFAULT_ITEM_COL])
        for u in range(self.n_users):
            index = list(np.where(user_np == u)[0])
            i = item_np[index]
            for item in i:
                R[u, item] = 1
        R = R.tolil()
        adj_mat[: self.n_users, self.n_users :] = R
        adj_mat[self.n_users :, : self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print("already create adjacency matrix", adj_mat.shape)
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        print("already normalize adjacency matrix")
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def get_graph_embeddings(self, config):
        """Get the graph embedding, if not previously stored then call the function to create.

        This method is for LCFN model.

        Returns:
            eigsh of the graph matrix
        """
        process_file_name = (
            "lcfn_"
            + config["dataset"]["dataset"]
            + "_"
            + config["dataset"]["data_split"]
            + (
                ("_" + str(config["dataset"]["percent"] * 100))
                if "percent" in config
                else ""
            )
        )
        process_path = os.path.join(
            config["system"]["process_dir"],
            config["dataset"]["dataset"] + "/",
        )
        process_file_name = os.path.join(process_path, process_file_name)
        ensureDir(process_file_name)
        print(process_file_name)

        try:

            with open(process_file_name + "/graph_embeddings.json") as f:
                line = f.readline()
                graph_embeddings = json.loads(line)
            f.close()

            print("already load graph embeddings")
        except Exception:
            graph_embeddings = self.create_graph_embeddings(config)

            f = open(process_file_name + "/graph_embeddings.json", "w")
            jsObj = json.dumps(graph_embeddings)
            f.write(jsObj)
            f.close()

        cut_off = config["model"]["cut_off"]
        [graph_u, graph_i] = graph_embeddings
        graph_u = np.array(graph_u)[:, 0 : int(cut_off * self.n_users)].astype(
            np.float32
        )
        graph_i = np.array(graph_i)[:, 0 : int(cut_off * self.n_items)].astype(
            np.float32
        )
        return [graph_u, graph_i]

    def create_graph_embeddings(self, config):
        """Create graph embeddings from the user and item hypergraph."""
        cut_off = config["model"]["cut_off"]
        user_np = np.array(self.ratings[DEFAULT_USER_COL])
        item_np = np.array(self.ratings[DEFAULT_ITEM_COL])
        user_number = self.n_users
        item_number = self.n_items
        tolerant = 0.1 ** 5
        epsilon = 0.1 ** 10
        H_u = sp.lil_matrix((user_number, item_number))
        H_v = sp.lil_matrix((item_number, user_number))
        D_u = sp.lil_matrix((user_number, user_number))
        D_v = sp.lil_matrix((item_number, item_number))
        I_u = sp.lil_matrix(np.eye(user_number, user_number))
        I_v = sp.lil_matrix(np.eye(item_number, item_number))

        for user in range(self.n_users):
            index = list(np.where(user_np == user)[0])
            i = item_np[index]
            for item in i:
                H_u[user, item] = 1
                H_v[item, user] = 1
                D_u[user, user] += 1
                D_v[item, item] += 1

        print("   constructing user matrix...")
        D_n = sp.lil_matrix((user_number, user_number))
        D_e = sp.lil_matrix((item_number, item_number))
        for i in range(user_number):
            D_n[i, i] = 1.0 / max(np.sqrt(D_u[i, i]), epsilon)
        for i in range(item_number):
            D_e[i, i] = 1.0 / max(D_v[i, i], epsilon)
        L_u = I_u - D_n * H_u * D_e * H_u.T * D_n

        print("   constructing item matrix...")
        D_n = sp.lil_matrix((item_number, item_number))
        D_e = sp.lil_matrix((user_number, user_number))
        for i in range(item_number):
            D_n[i, i] = 1.0 / max(np.sqrt(D_v[i, i]), epsilon)
        for i in range(user_number):
            D_e[i, i] = 1.0 / max(D_u[i, i], epsilon)
        L_v = I_v - D_n * H_v * D_e * H_v.T * D_n

        print("Decomposing the laplacian matrices...")
        print("   decomposing user matrix...")

        [Lamda, user_graph_embeddings] = eigsh(
            L_u, k=int(cut_off * self.n_users), which="SM", tol=tolerant
        )
        print(Lamda[0:10])
        print("   decomposing item matrix...")
        [Lamda, item_graph_embeddings] = eigsh(
            L_v, k=int(cut_off * self.n_items), which="SM", tol=tolerant
        )
        print(Lamda[0:10])

        return [user_graph_embeddings.tolist(), item_graph_embeddings.tolist()]

import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from collections import defaultdict
from itertools import count
import pandas as pd


class Data(object):
# Data utilit class for the NGCF
    def __init__(self, path,train, test, vad, batch_size):
        self.batch_size = batch_size
        self.path = path
        self.df = train
        self.n_users, self.n_items = 0, 0
        self.n_train = 0
        self.neg_pools = {}

        self.uid_map = defaultdict(count().__next__)
        self.iid_map = defaultdict(count().__next__)
        self.uids = np.array([self.uid_map[uid] for uid in self.df["col_user"].values], dtype=np.int32)
        self.iids = np.array([self.iid_map[iid] for iid in self.df["col_item"].values], dtype=np.int32)
        self.exist_users = list(set(self.uids))
        self.n_users = len(set(self.uids))
        self.n_items = len(set(self.iids))

        self.print_statistics()
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.train_items = {}

        for u in self.exist_users:
            index = list(np.where(self.uids == u)[0])
            i = self.iids[index]
            self.train_items[u] = i
            for item in i:
                self.R[u,item] = 1
                self.n_train += 1

        self.test = self.reindex(test)
        self.vad = self.reindex(vad)


    def reindex(self,df):
    """
    """
        users = df["col_user"]
        items = df["col_item"]
        re_uids = []
        re_iids = []
        for i in range(len(users)):
            if self.uid_map[users[i]] <=self.n_users and self.iid_map[items[i]] <=self.n_items:
                re_uids.append(self.uid_map[users[i]])
                re_iids.append(self.iid_map[items[i]])
        ratings = np.ones(len(re_uids))

        return pd.DataFrame(list(zip(re_uids,re_iids,ratings)),columns=['col_user','col_item','col_rating'])

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        return users, pos_items, neg_items

    def sample_all_users_pos_items(self):
        self.all_train_users = []

        self.all_train_pos_items = []
        for u in self.exist_users:
            self.all_train_users += [u] * len(self.train_items[u])
            self.all_train_pos_items += self.train_items[u]

    def epoch_sample(self):
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        neg_items = []
        for u in self.all_train_users:
            neg_items += sample_neg_items_for_u(u,1)

        perm = np.random.permutation(len(self.all_train_users))
        users = np.array(self.all_train_users)[perm]
        pos_items = np.array(self.all_train_pos_items)[perm]
        neg_items = np.array(neg_items)[perm]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_train=%d' % (self.n_train))




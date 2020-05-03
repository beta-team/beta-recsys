import os
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.sparse import coo_matrix
from beta_rec.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
)
from beta_rec.utils.common_util import get_random_rep
from beta_rec.utils.aliasTable import AliasTable
from beta_rec.utils.triple_sampler import Sampler
from beta_rec.datasets.data_load import load_split_dataset, load_item_fea_dic


pd.options.mode.chained_assignment = None  # default='warn'


def intersect_train_test(train, test):
    """ Get the intersect lists of users and items that exist in both train and test

    Args:
        train (DataFrame):
        test (DataFrame):

    Returns:
        users (list): users list
        items (list): items list

    """
    users = list(
        set(train[DEFAULT_USER_COL].unique().flatten()).intersection(
            set(test[DEFAULT_USER_COL].unique().flatten())
        )
    )
    items = list(
        set(train[DEFAULT_ITEM_COL].unique().flatten()).intersection(
            set(test[DEFAULT_ITEM_COL].unique().flatten())
        )
    )
    return users, items


class Dataset(object):
    """
        Base Dataset class for all the model
    """

    def __init__(self, config):
        """Constructor

        Args:
            config:
        """
        self.config = config
        self.n_users = 0
        self.n_items = 0
        self.sub_set = 0
        self.random_dim = 512
        # subset of the dataset. use a small set of users and itesm if >0, otherwise use full dataset
        if "sub_set" in config:
            self.sub_set = config["sub_set"]
        if "random_dim" in config:
            self.random_dim = config["random_dim"]
        # data preprocessing for training and test data
        # To be replaced with new data method
        train, valid, test = load_split_dataset(config)
        self.train = self._intersect_train_test(train, test[0])
        self.valid = self._reindex_list(valid)
        self.test = self._reindex_list(test)
        self.item_sampler = AliasTable(
            self.train[DEFAULT_ITEM_COL].value_counts().to_dict()
        )
        self.user_sampler = AliasTable(
            self.train[DEFAULT_USER_COL].value_counts().to_dict()
        )

    def sample_triple(self, dump=True, load_save=True):
        """
        Sample triples or load triples samples from files. Only applicable for basket based Recommender
        Returns:
            None

        """
        # need to be specified if need samples
        self.config["sample_dir"] = os.path.join(
            self.config["root_dir"], self.config["sample_dir"]
        )

        sample_file = (
            self.config["sample_dir"]
            + "triple_"
            + self.config["dataset"]
            + "_"
            + str(self.config["percent"] * 100)
            + "_"
            + str(self.config["n_sample"])
            + "_"
            + str(self.config["temp_train"])
            + ".csv"
        )
        my_sampler = Sampler(
            self.train,
            sample_file,
            self.config["n_sample"],
            dump=dump,
            load_save=load_save,
        )
        return my_sampler.sample()

    def generate_train_data(self):
        """ Generate a rating matrix for interactions

        Returns:
            (sigma_matrix, rating_matrix)
            sigma_matrix with rating being 1
            rating_matrix with rating being the real rating
        """
        train_data = (
            self.train.groupby(["col_user", "col_item"])
            .sum()
            .to_frame("col_rating")
            .reset_index()
        )
        n = self.n_users
        m = self.n_items
        user_record = train_data.col_user.tolist()
        movie_record = train_data.col_item.tolist()
        ratings_record = train_data.col_rating.tolist()
        rating_matrix = np.zeros([n, m])
        sigma_matrix = np.zeros([n, m])
        for i in range(len(user_record)):
            rating_matrix[user_record[i] - 1, movie_record[i] - 1] = ratings_record[i]
            sigma_matrix[user_record[i] - 1, movie_record[i] - 1] = 1
        # load test_data
        print("data load finish")
        return sigma_matrix, rating_matrix

    def generate_sparse_train_data(self):
        """Generate a sparse matrix for interactions.

        Returns:
            coo_matrix
        """

        train_data = (
            self.train.groupby(["col_user", "col_item"])
            .sum()
            .to_frame("col_rating")
            .reset_index()
        )
        n = self.n_users
        m = self.n_items
        user_record = train_data.col_user.tolist()
        movie_record = train_data.col_item.tolist()
        ratings_record = train_data.col_rating.tolist()
        print("data load finish")
        return coo_matrix((ratings_record, (user_record, movie_record)))

    def _intersect_train_test(self, train, test, implicit=True):
        """ process the dataset to reindex userID and itemID, also set rating as implicit feedback

        Parameters:
            train (pandas.DataFrame): training data with at least columns (col_user, col_item, col_rating) 
            test (pandas.DataFrame): test data with at least columns (col_user, col_item, col_rating)
                    test can be None, if so, we only process the training data
            implicit (bool): if true, set rating>0 to rating = 1 

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed.
        
        """
        users_ser, items_ser = intersect_train_test(train, test)
        self.n_users = -1
        self.n_items = -1
        while self.n_users != len(users_ser) or self.n_items != len(items_ser):
            test = test[
                test[DEFAULT_USER_COL].isin(users_ser)
                & test[DEFAULT_ITEM_COL].isin(items_ser)
            ]
            train = train[
                train[DEFAULT_USER_COL].isin(users_ser)
                & train[DEFAULT_ITEM_COL].isin(items_ser)
            ]
            self.n_users = train[DEFAULT_USER_COL].nunique()
            self.n_items = train[DEFAULT_ITEM_COL].nunique()
            users_ser, items_ser = intersect_train_test(train, test)
        print(f"After intersection, train statistics")
        print(
            tabulate(
                train.agg(["count", "nunique"]),
                headers=test.columns,
                tablefmt="psql",
                disable_numparse=True,
            )
        )
        print(f"After intersection, test statistics")
        print(
            tabulate(
                test.agg(["count", "nunique"]),
                headers=test.columns,
                tablefmt="psql",
                disable_numparse=True,
            )
        )
        self.user_pool = users_ser
        self.item_pool = items_ser

        # Reindex user and item index
        self.user2id = dict(zip(np.array(self.user_pool), np.arange(self.n_users)))
        self.id2user = {self.user2id[k]: k for k in self.user2id}

        self.item2id = dict(zip(np.array(self.item_pool), np.arange(self.n_items)))
        self.id2item = {self.item2id[k]: k for k in self.item2id}

        if self.sub_set > 0:  # for small dataset
            print("using a subset of the dataset:", self.sub_set)
            self.user_pool = self.user_pool[: self.sub_set]
            self.item_pool = self.item_pool[: self.sub_set]
        return self._reindex(train, implicit)

    def _reindex_list(self, df_list):
        """_reindex for list of dataset. For example, validate and test can be a list for evaluation
        
        """
        df_list_new = []
        for df in df_list:
            df = df[
                df[DEFAULT_USER_COL].isin(self.user_pool)
                & df[DEFAULT_ITEM_COL].isin(self.item_pool)
            ]
            df_list_new.append(self._reindex(df))
        return df_list_new

    def _reindex(self, df, implicit=True):
        """ 
        Process dataset to reindex userID and itemID, also set rating as implicit feedback

        Parameters:
            df (pandas.DataFrame): dataframe with at least columns (col_user, col_item, col_rating) 
            implicit (bool): if true, set rating>0 to rating = 1 

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed.
        
        """

        # If testing dataset is None
        if df is None:
            return None

        # Map user_idx and item_idx
        df[DEFAULT_USER_COL] = df[DEFAULT_USER_COL].apply(lambda x: self.user2id[x])
        df[DEFAULT_ITEM_COL] = df[DEFAULT_ITEM_COL].apply(lambda x: self.item2id[x])

        # If implicit feedback, set rating as 1.0 or 0.0
        if implicit:
            df[DEFAULT_RATING_COL] = df[DEFAULT_RATING_COL].apply(
                lambda x: float(x > 0)
            )

        # Select relevant columns
        #         df_reindex = df[
        #             [DEFAULT_USER_COL + "_idx", DEFAULT_ITEM_COL + "_idx", DEFAULT_RATING_COL]
        #         ]
        #         df_reindex.columns = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]

        return df

    def init_item_fea(self, config):
        """
        initialize item feature

        """
        if "item_fea_type" in config:
            fea_type = config["item_fea_type"]
        else:
            fea_type = "random"
        data_str = config["dataset"]
        print(
            "Loading item feature for dataset:", data_str, " type:", fea_type,
        )

        if fea_type == "random":
            self.item_feature = get_random_rep(self.n_items, self.random_dim)
        elif fea_type == "one_hot":
            item_fea_dic = load_item_fea_dic(config, fea_type="one_hot")
            self.item_feature = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
        elif fea_type == "word2vec":
            item_fea_dic = load_item_fea_dic(config, fea_type="word2vec")
            self.item_feature = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
        elif fea_type == "bert":
            item_fea_dic = load_item_fea_dic(config, fea_type="bert")
            self.item_feature = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
        elif fea_type == "random_one_hot" or fea_type == "one_hot_random":
            rand_item_fea = get_random_rep(self.n_items, 512)
            item_fea_dic = load_item_fea_dic(config, fea_type="one_hot")
            item_fea_list = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate((item_fea_list, rand_item_fea), axis=1)
        elif fea_type == "random_cate" or fea_type == "cate_random":
            rand_item_fea = get_random_rep(self.n_items, 512)
            item_fea_dic = load_item_fea_dic(config, fea_type="cate")
            item_fea_list = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate((item_fea_list, rand_item_fea), axis=1)
        elif fea_type == "random_word2vec" or fea_type == "word2vec_random":
            rand_item_fea = get_random_rep(self.n_items, 512)
            item_fea_dic = load_item_fea_dic(config, fea_type="word2vec")
            item_fea_list = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate((item_fea_list, rand_item_fea), axis=1)
        elif fea_type == "random_bert" or fea_type == "bert_random":
            rand_item_fea = get_random_rep(self.n_items, 512)
            item_fea_dic = load_item_fea_dic(config, fea_type="bert")
            item_fea_list = np.array(
                [item_fea_dic[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate((item_fea_list, rand_item_fea), axis=1)
        elif fea_type == "word2vec_one_hot" or fea_type == "one_hot_word2vec":
            item_fea_dic1 = load_item_fea_dic(config, fea_type="one_hot")
            item_fea_dic2 = load_item_fea_dic(config, fea_type="word2vec")
            item_fea_list1 = np.array(
                [item_fea_dic1[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            item_fea_list2 = np.array(
                [item_fea_dic2[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate((item_fea_list1, item_fea_list2), axis=1)
        elif (
            fea_type == "word2vec_one_hot_random"
            or fea_type == "random_one_hot_word2vec"
            or fea_type == "one_hot_random_word2vec"
            or fea_type == "random_word2vec_one_hot"
        ):
            rand_item_fea = get_random_rep(self.n_items, 512)
            item_fea_dic1 = load_item_fea_dic(config, fea_type="one_hot")
            item_fea_dic2 = load_item_fea_dic(config, fea_type="word2vec")
            item_fea_list1 = np.array(
                [item_fea_dic1[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            item_fea_list2 = np.array(
                [item_fea_dic2[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate(
                (item_fea_list1, item_fea_list2, rand_item_fea), axis=1
            )
        elif (
            fea_type == "word2vec_one_hot_bert"
            or fea_type == "bert_one_hot_word2vec"
            or fea_type == "one_hot_bert_word2vec"
        ):
            item_fea_dic1 = load_item_fea_dic(config, fea_type="one_hot")
            item_fea_dic2 = load_item_fea_dic(config, fea_type="word2vec")
            item_fea_dic3 = load_item_fea_dic(config, fea_type="bert")
            item_fea_list1 = np.array(
                [item_fea_dic1[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            item_fea_list2 = np.array(
                [item_fea_dic2[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            item_fea_list3 = np.array(
                [item_fea_dic3[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate(
                (item_fea_list1, item_fea_list2, item_fea_list3), axis=1
            )
        elif (
            fea_type == "random_word2vec_one_hot_bert"
            or fea_type == "bert_one_hot_word2vec_random"
            or fea_type == "one_hot_random_bert_word2vec"
            or fea_type == "one_hot_bert_random_word2vec"
            or fea_type == "one_hot_word2vec_bert_random"
            or fea_type == "random_one_hot_word2vec_bert"
        ):
            rand_item_fea = get_random_rep(self.n_items, 512)
            item_fea_dic1 = load_item_fea_dic(config, fea_type="one_hot")
            item_fea_dic2 = load_item_fea_dic(config, fea_type="word2vec")
            item_fea_dic3 = load_item_fea_dic(config, fea_type="bert")
            item_fea_list1 = np.array(
                [item_fea_dic1[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            item_fea_list2 = np.array(
                [item_fea_dic2[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            item_fea_list3 = np.array(
                [item_fea_dic3[self.id2item[k]] for k in np.arange(self.n_items)]
            )
            self.item_feature = np.concatenate(
                (item_fea_list1, item_fea_list2, item_fea_list3, rand_item_fea), axis=1
            )
        else:
            print(
                "[ERROR]: CANNOT support feature type {}! intialize with random feature".format(
                    fea_type
                )
            )
            self.item_feature = get_random_rep(self.n_items, self.random_dim)

    def init_user_fea(self):
        """
        initialize user feature

        """
        if "user_fea_type" in self.config:
            fea_type = self.config["user_fea_type"]
        else:
            fea_type = "random"

        print(
            "init user featrue for dataset:", self.config["dataset"], " type:", fea_type
        )
        if fea_type == "random":
            self.user_feature = get_random_rep(self.n_users, self.random_dim)
        else:
            print(
                "[ERROR]: CANNOT support other feature type, use 'random' user featrue instead!"
            )
            self.user_feature = get_random_rep(self.n_users, self.random_dim)
            # load_user_fea(config)

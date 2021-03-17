import numpy as np

from ..datasets.data_load import load_item_fea_dic
from ..utils.common_util import get_random_rep


class Auxiliary(object):
    r"""A Auxiliary Data object, which is able read various feature for users and items.

    Args:
        config (dict): configs dict.

    """

    def __init__(self, config, n_users, n_items):
        """Initialize Auxiliary Class."""
        self.config = config
        self.n_users = n_users
        self.n_items = n_items
        self.random_dim = 512
        # subset of the dataset. use a small set of users and items if >0, otherwise use full dataset
        if config and "random_dim" in config:
            self.random_dim = config["model"]["random_dim"]
        # self.init_item_fea()
        # self.init_user_fea()

    def init_item_fea(self):
        """Initialize item feature."""
        config = self.config
        if config and "item_fea_type" in config["dataset"]:
            fea_type = config["dataset"]["item_fea_type"]
        else:
            fea_type = "random"
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
        """Initialize user feature for VBCAR model."""
        if self.config and "user_fea_type" in self.config["dataset"]:
            fea_type = self.config["dataset"]["user_fea_type"]
        else:
            fea_type = "random"

        if fea_type == "random":
            self.user_feature = get_random_rep(self.n_users, self.random_dim)
        else:
            print(
                "[ERROR]: CANNOT support other feature type, use 'random' user featrue instead!"
            )
            self.user_feature = get_random_rep(self.n_users, self.random_dim)

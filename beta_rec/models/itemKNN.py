import torch
import torch.nn as nn
from torch.nn import Parameter

from beta_rec.models.torch_engine import ModelEngine
from beta_rec.utils.common_util import print_dict_as_table, timeit

import scipy.sparse as ssp
import numpy as np


def top_k(values, k, exclude=[]):
    '''
    Return the indices of the k items with the highest value in the list of values.
    Exclude the ids from the list "exclude".
    '''
    # Put low similarity to viewed items to exclude them from recommendations
    values[exclude] = -np.inf
    return list(np.argpartition(-values, range(k))[:k])


def get_sparse_vector(ids, length, values=None):
    '''
    Converts a list of ids into a sparse vector of length "length" where the elements
    corresponding to the ids are given the values in "values". If "values" is None, the elements are set to 1.
    '''
    n = len(ids)
    if values is None:
        return ssp.coo_matrix((np.ones(n), (ids, np.zeros(n))), (length, 1)).tocsc()
    else:
        return ssp.coo_matrix((values, (ids, np.zeros(n))), (length, 1)).tocsc()


class UserKNN(torch.nn.Module):
    """A PyTorch Module for UserKNN model."""

    def __init__(self, config):
        """Initialize UserKNN Class."""
        super(UserKNN, self).__init__()
        self.config = config
        self.device = self.config["device_str"]
        self.n_users = self.config["n_users"]
        self.n_items = self.config["n_items"]
        self.neighbourhood_size = self.config['neighbourhood_size']

    def prepare_model(self, data):
        row = data.train['col_user'].to_numpy()
        col = data.train['col_item'].to_numpy()
        print(row.shape)
        self.binary_user_item = ssp.coo_matrix((np.ones(len(data.train)),
                                               (row, col)), shape=(self.n_users, self.n_items)).tocsr()
        print(self.binary_user_item.shape)

    def _items_count_per_user(self):
        if not hasattr(self, '__items_count_per_user'):
            self.__items_count_per_user = np.asarray(self.binary_user_item.sum(axis=1)).ravel()
        return self.__items_count_per_user

    def similarity_with_users(self, sequence):
        '''
            Calculate the similarity between the a given user and all users according to the overlap ratio.
        :param sequence: the user's interacted items
        :return:
        '''
        sparse_sequence = get_sparse_vector(sequence, self.n_items)
        overlap = self.binary_user_item.dot(sparse_sequence).toarray().ravel()
        overlap[overlap != 0] /= np.sqrt(self._items_count_per_user()[overlap != 0])
        return overlap

    def forward(self, batch_data):
        """UserKNN is a non-personalised model. The forward function is not needed.

        Args:
            batch_data: tuple consists of (users, pos_items, neg_items), which must be LongTensor.
        """
        return 0.0

    def predict(self, users, items):
        """Predict result with the model.

        Args:
            users (int, or list of int):  user id(s).
            items (int, or list of int):  item id(s).
        Return:
            scores (int, or list of int): predicted scores of these user-item pairs.
        """
        scores = []
        for i in range(len(users)):
            sequence = self.binary_user_item.getcol(users[i]).nonzero()[0]
            # sparse_sequence = self.binary_user_item.getcol(users[i])
            sim_with_users = self.similarity_with_users(sequence)
            nearest_neighbour = top_k(sim_with_users, self.neighbourhood_size)
            neighbour_items = get_sparse_vector(nearest_neighbour, self.n_users,
                                                values=sim_with_users[nearest_neighbour])
            sim_with_items = self.binary_user_item.T.dot(neighbour_items).toarray().ravel()
            sim_with_items[sequence] = -np.inf
            scores.append(sim_with_items[items[i]])
        return torch.tensor(scores)


class UserKNNEngine(ModelEngine):
    """UserKNNEngine Class."""

    def __init__(self, config):
        """Initialize UserKNNEngine Class."""
        print("userKNNEngine init")
        self.config = config
        self.model = UserKNN(config["model"])
        # super(UserKNNEngine, self).__init__(config)

    def train_single_batch(self, batch_data):
        """Train a single batch.
            However, userKNN is a neighbourhood model bases its prediction on the similarity relationships among users.
            It requires no training procedure.
        Args:
            batch_data (list): batch users, positive items and negative items.
        Return:
            0
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        return 0

    @timeit
    def train_an_epoch(self, train_loader, epoch_id):
        """Train a epoch, generate batch_data from data_loader, and call train_single_batch.
            Like the train_single_batch method, UserKNN requires no training procedure.

        Args:
            train_loader (DataLoader):
            epoch_id (int): set to 1.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        # self.model.train()
        print(f"[Training Epoch {epoch_id}] skipped")
        self.writer.add_scalar("model/loss", 0.0, epoch_id)
        self.writer.add_scalar("model/regularizer", 0.0, epoch_id)

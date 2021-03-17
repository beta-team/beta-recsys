import numpy as np
import scipy.sparse as ssp
import torch

from beta_rec.models.torch_engine import ModelEngine
from beta_rec.utils.common_util import timeit


def top_k(values, k, exclude=[]):
    """Return the indices of the k items with the highest value in the list of values.

    Exclude the ids from the list "exclude".
    :param values:
    :param k:
    :param exclude:
    :return:
    """
    # Put low similarity to viewed items to exclude them from recommendations
    values[exclude] = -np.inf
    return list(np.argpartition(-values, range(k))[:k])


def get_sparse_vector(ids, length, values=None):
    """Sparse vector generation.

    If "values" is None, the elements are set to 1.
    :param ids:
    :param length:
    :param values:
    :return:
    """
    n = len(ids)
    if values is None:
        return ssp.coo_matrix((np.ones(n), (ids, np.zeros(n))), (length, 1)).tocsc()
    else:
        return ssp.coo_matrix((values, (ids, np.zeros(n))), (length, 1)).tocsc()


class ItemKNN(torch.nn.Module):
    """A PyTorch Module for ItemKNN model."""

    def __init__(self, config):
        """Initialize ItemKNN Class."""
        super(ItemKNN, self).__init__()
        self.config = config
        self.device = self.config["device_str"]
        self.n_users = self.config["n_users"]
        self.n_items = self.config["n_items"]
        self.neighbourhood_size = self.config["neighbourhood_size"]

    def generate_item_sim_matrix(self):
        """Calculate the similarity matrix between items.

        :return: item_sim_matrix
        """
        item_sim_matrix = np.ones((self.n_items, self.n_items), dtype=np.float32)
        for item in range(self.n_items):
            user_sequence = self.binary_user_item.getcol(item).nonzero()[0]
            item_sim_matrix[item] = self.similarity_with_items(user_sequence)
        return item_sim_matrix

    def prepare_model(self, data):
        """Load data into matrices.

        :param data:
        :return:
        """
        row = data.train["col_user"].to_numpy()
        col = data.train["col_item"].to_numpy()
        self.binary_user_item = ssp.coo_matrix(
            (np.ones(len(data.train)), (row, col)), shape=(self.n_users, self.n_items)
        ).tocsr()
        self.item_sim_matrix = self.generate_item_sim_matrix()

    def _users_count_per_item(self):
        """Calculate the number of interacted users for an item.

        :return:
        """
        if not hasattr(self, "__users_count_per_item"):
            self.__users_count_per_item = np.asarray(
                self.binary_user_item.sum(axis=0)
            ).ravel()
        return self.__users_count_per_item

    def similarity_with_items(self, sequence):
        """Calculate the similarity between the a given item and all items according to the overlap ratio.

        :param sequence: users that interacted with a given item
        :return:
        """
        sparse_sequence = get_sparse_vector(sequence, self.n_users)
        overlap = self.binary_user_item.T.dot(sparse_sequence).toarray().ravel()
        overlap[overlap != 0] /= np.sqrt(self._users_count_per_item()[overlap != 0])
        return overlap

    def forward(self, batch_data):
        """Redundant method for ItemKNN.

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
            sequence = self.binary_user_item.getrow(users[i]).nonzero()[0]
            for idx, item in enumerate(sequence):
                sub_sim_with_items = self.item_sim_matrix[item]
                if idx == 0:
                    sim_with_items = sub_sim_with_items
                else:
                    sim_with_items = np.add(sim_with_items, self.item_sim_matrix[item])
            scores.append(sim_with_items[items[i]])
        return torch.tensor(scores)


class ItemKNNEngine(ModelEngine):
    """UserKNNEngine Class."""

    def __init__(self, config):
        """Initialize UserKNNEngine Class."""
        self.config = config
        self.model = ItemKNN(config["model"])
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

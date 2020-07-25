import torch
import torch.nn as nn
from torch.nn import Parameter

from beta_rec.models.torch_engine import ModelEngine
from beta_rec.utils.common_util import print_dict_as_table, timeit


class MF(torch.nn.Module):
    """A pytorch Module for Matrix Factorization."""

    def __init__(self, config):
        """Initialize MF Class."""
        super(MF, self).__init__()
        self.config = config
        self.device = self.config["device_str"]
        self.stddev = self.config["stddev"] if "stddev" in self.config else 0.1
        self.n_users = self.config["n_users"]
        self.n_items = self.config["n_items"]
        self.emb_dim = self.config["emb_dim"]
        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        self.global_bias = Parameter(torch.zeros(1))
        self.user_bias.weight.data.fill_(0.0)
        self.item_bias.weight.data.fill_(0.0)
        self.global_bias.data.fill_(0.0)
        nn.init.normal_(self.user_emb.weight, 0, self.stddev)
        nn.init.normal_(self.item_emb.weight, 0, self.stddev)

    def forward(self, batch_data):
        """Trian the model.

        Args:
            batch_data: tuple consists of (users, pos_items, neg_items), which must be LongTensor.
        """
        users, items = batch_data
        u_emb = self.user_emb(users)
        u_bias = self.user_bias(users)
        i_emb = self.item_emb(items)
        i_bias = self.item_bias(items)
        scores = torch.sigmoid(
            torch.sum(torch.mul(u_emb, i_emb).squeeze(), dim=1)
            + u_bias.squeeze()
            + i_bias.squeeze()
            + self.global_bias
        )
        regularizer = (
            (u_emb ** 2).sum()
            + (i_emb ** 2).sum()
            + (u_bias ** 2).sum()
            + (i_bias ** 2).sum()
        ) / u_emb.size()[0]
        return scores, regularizer

    def predict(self, users, items):
        """Predcit result with the model.

        Args:
            users (int, or list of int):  user id(s).
            items (int, or list of int):  item id(s).
        Return:
            scores (int, or list of int): predicted scores of these user-item pairs.
        """
        users_t = torch.LongTensor(users).to(self.device)
        items_t = torch.LongTensor(items).to(self.device)
        with torch.no_grad():
            scores, _ = self.forward((users_t, items_t))
        return scores


class MFEngine(ModelEngine):
    """MFEngine Class."""

    def __init__(self, config):
        """Initialize MFEngine Class."""
        self.config = config
        print_dict_as_table(config["model"], tag="MF model config")
        self.model = MF(config["model"])
        self.reg = (
            config["model"]["reg"] if "reg" in config else 0.0
        )  # the regularization coefficient.
        self.batch_size = config["model"]["batch_size"]
        super(MFEngine, self).__init__(config)
        self.model.to(self.device)
        self.loss = (
            self.config["model"]["loss"] if "loss" in self.config["model"] else "bpr"
        )
        print(f"using {self.loss} loss...")

    def train_single_batch(self, batch_data):
        """Train a single batch.

        Args:
            batch_data (list): batch users, positive items and negative items.
        Return:
            loss (float): batch loss.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        if self.loss == "bpr":
            users, pos_items, neg_items = batch_data
            pos_scores, pos_regularizer = self.model.forward((users, pos_items))
            neg_scores, neg_regularizer = self.model.forward((users, neg_items))
            loss = self.bpr_loss(pos_scores, neg_scores)
            regularizer = pos_regularizer + neg_regularizer
        elif self.loss == "bce":
            users, items, ratings = batch_data
            scores, regularizer = self.model.forward((users, items))
            loss = self.bce_loss(scores, ratings)
        else:
            raise RuntimeError(
                f"Unsupported loss type {self.loss}, try other options: 'bpr' or 'bce'"
            )
        batch_loss = loss + self.reg * regularizer
        batch_loss.backward()
        self.optimizer.step()
        return loss.item(), regularizer.item()

    @timeit
    def train_an_epoch(self, train_loader, epoch_id):
        """Train a epoch, generate batch_data from data_loader, and call train_single_batch.

        Args:
            train_loader (DataLoader):
            epoch_id (int): the number of epoch.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0.0
        regularizer = 0.0
        for batch_data in train_loader:
            loss, reg = self.train_single_batch(batch_data)
            total_loss += loss
            regularizer += reg
        print(f"[Training Epoch {epoch_id}], Loss {loss}, Regularizer {regularizer}")
        self.writer.add_scalar("model/loss", total_loss, epoch_id)
        self.writer.add_scalar("model/regularizer", regularizer, epoch_id)

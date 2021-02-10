import numpy as np
import torch
import torch.nn as nn

from beta_rec.models.torch_engine import ModelEngine


class LCFN(torch.nn.Module):
    """Model initialisation, embedding generation and prediction of Low-pass Collaborative Filters."""

    def __init__(self, config, graph_embeddings):
        """Initialize LightGCN Class."""
        super(LCFN, self).__init__()
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        [self.P, self.Q] = graph_embeddings
        self.layer = config["layer"]
        self.lamda = config["lamda"]
        self.frequence_user = int(np.shape(self.P)[1])
        self.frequence_item = int(np.shape(self.Q)[1])
        self.P = torch.tensor(self.P)
        self.Q = torch.tensor(self.Q)
        self.user_embeddings = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embeddings = nn.Embedding(self.n_items, self.emb_dim)
        self.init_emb()

        self.user_filters = []
        for k in range(self.layer):
            self.user_filters.append(
                torch.normal(mean=1, std=0.001, size=(1, self.frequence_user))[0]
            )
        self.item_filters = []
        for k in range(self.layer):
            self.item_filters.append(
                torch.normal(size=(1, self.frequence_item), mean=1, std=0.001)[0]
            )

        self.transformers = []
        for k in range(self.layer):
            self.transformers.append(
                torch.tensor(
                    (
                        np.random.normal(0, 0.001, (self.emb_dim, self.emb_dim))
                        + np.diag(np.random.normal(1, 0.001, self.emb_dim))
                    ).astype(np.float32)
                )
            )

    def init_emb(self):
        """Initialize users and items' embeddings."""
        nn.init.normal(self.user_embeddings.weight, mean=0.01, std=0.02)
        nn.init.normal(self.item_embeddings.weight, mean=0.01, std=0.02)

    def forward(self):
        """Graph propagation and noise filtering."""
        self.P = self.P.to(self.device)
        self.Q = self.Q.to(self.device)
        for i in range(self.layer):
            self.user_filters[i] = self.user_filters[i].to(self.device)
            self.item_filters[i] = self.item_filters[i].to(self.device)
            self.transformers[i] = self.transformers[i].to(self.device)
        User_embedding = self.user_embeddings
        self.user_all_embeddings = [User_embedding.weight]
        for k in range(self.layer):
            User_embedding = torch.matmul(
                torch.matmul(self.P, torch.diag(self.user_filters[k])),
                torch.matmul(torch.transpose(self.P, 0, 1), User_embedding.weight),
            )
            User_embedding = torch.sigmoid(
                torch.matmul(User_embedding, self.transformers[k])
            )
            self.user_all_embeddings += [User_embedding]
        self.user_all_embeddings = torch.cat(self.user_all_embeddings, 1)

        Item_embedding = self.item_embeddings.to(self.device)
        self.item_all_embeddings = [Item_embedding.weight]
        for k in range(self.layer):
            Item_embedding = torch.matmul(
                torch.matmul(self.Q, torch.diag(self.item_filters[k])),
                torch.matmul(torch.transpose(self.Q, 0, 1), Item_embedding.weight),
            )
            Item_embedding = torch.sigmoid(
                torch.matmul(Item_embedding, self.transformers[k])
            )
            self.item_all_embeddings += [Item_embedding]
        self.item_all_embeddings = torch.cat(self.item_all_embeddings, 1)

    def predict(self, users, items):
        """Model prediction."""
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)

        with torch.no_grad():
            u_embeddings = self.user_all_embeddings[users_t]
            i_embeddings = self.item_all_embeddings[items_t]
            scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

        return scores


class LCFNEngine(ModelEngine):
    """LightGCNEngine Class."""

    # A class includes train an epoch and train a batch of Low-pass collaborative filtering

    def __init__(self, config):
        """Initialize LightGCNEngine Class."""
        self.config = config
        self.graph_embeddings = config["model"]["graph_embeddings"]
        self.model = LCFN(config["model"], self.graph_embeddings)
        super(LCFNEngine, self).__init__(config)
        self.model.to(self.device)

    def train_single_batch(self, batch_data):
        """Train the model in a single batch.

        Args:
            batch_data (list): batch users, positive items and negative items.
        Return:
            loss (float): batch loss.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        self.model.forward()
        user_embeddings = self.model.user_embeddings
        item_embeddings = self.model.item_embeddings
        user_all_embeddings = self.model.user_all_embeddings
        item_all_embeddings = self.model.item_all_embeddings

        batch_users, pos_items, neg_items = batch_data

        u_embedding_loss = user_embeddings(batch_users)
        pos_i_embeddings_loss = item_embeddings(pos_items)
        neg_i_embeddings_loss = item_embeddings(neg_items)
        user_all_emb = user_all_embeddings[batch_users]
        pos_i_all = item_all_embeddings[pos_items]
        neg_i_all = item_all_embeddings[neg_items]

        batch_loss = self.loss_comput(
            u_embedding_loss,
            pos_i_embeddings_loss,
            neg_i_embeddings_loss,
            user_all_emb,
            pos_i_all,
            neg_i_all,
        )

        batch_loss.backward()
        self.optimizer.step()
        loss = batch_loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        """Train the model in one epoch.

        Args:
            epoch_id (int): the number of epoch.
            train_loader (function): user, pos_items and neg_items generator.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0.0

        for batch_data in train_loader:
            loss = self.train_single_batch(batch_data)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def loss_comput(
        self,
        u_embeddings_loss,
        pos_i_embeddings_loss,
        neg_i_embeddings_loss,
        user_all_emb,
        pos_i_all,
        neg_i_all,
    ):
        """Loss computation."""
        pos_scores = torch.sum(torch.mul(user_all_emb, pos_i_all), dim=1)
        neg_scores = torch.sum(torch.mul(user_all_emb, neg_i_all), dim=1)
        bpr_loss = self.bpr_loss(pos_scores, neg_scores)

        regularizer = (
            torch.norm(u_embeddings_loss)
            + torch.norm(pos_i_embeddings_loss)
            + torch.norm(neg_i_embeddings_loss)
        )

        layer = self.model.layer
        lamda = self.model.lamda

        for k in range(layer):
            regularizer += (
                torch.norm(self.model.user_filters[k])
                + torch.norm(self.model.item_filters[k])
                + torch.norm(self.model.transformers[k])
            )

        regularizer = lamda * regularizer

        loss = bpr_loss + regularizer

        return loss

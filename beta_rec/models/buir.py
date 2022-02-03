import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from beta_rec.models.torch_engine import ModelEngine


class BUIR_NB(nn.Module):
    def __init__(self, config):
        super(BUIR_NB, self).__init__()
        self.config = config
        self.user_count = config["n_users"]
        self.item_count = config["n_items"]
        self.latent_size = config["emb_dim"]
        self.momentum = config["momentum"]
        self.norm_adj = config["norm_adj"]
        n_layers = 3
        drop_flag = False
        self.online_encoder = LGCN_Encoder(
            self.user_count,
            self.item_count,
            self.latent_size,
            self.norm_adj,
            n_layers,
            drop_flag,
        )
        self.target_encoder = LGCN_Encoder(
            self.user_count,
            self.item_count,
            self.latent_size,
            self.norm_adj,
            n_layers,
            drop_flag,
        )

        self.predictor = nn.Linear(self.latent_size, self.latent_size)

        self._init_target()

    def _init_target(self):
        for param_o, param_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def _update_target(self):
        for param_o, param_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data = param_t.data * self.momentum + param_o.data * (
                1.0 - self.momentum
            )

    def forward(self, users, items):
        u_online, i_online = self.online_encoder(users, items)
        u_target, i_target = self.target_encoder(users, items)
        return self.predictor(u_online), u_target, self.predictor(i_online), i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.get_embedding()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def get_loss(self, output):
        u_online, u_target, i_online, i_target = output

        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)

        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)

        return (loss_ui + loss_iu).mean()

    def predict(self, users, items):
        """Predict result with the model.

        Args:
            users (int, or list of int):  user id.
            items (int, or list of int):  item id.
        Return:
            scores (int): dot product.
        """
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            u_online, u_target, i_online, i_target = self.get_embedding()
            u_online_t = u_online[users_t]
            u_target_t = u_target[users_t]
            i_online_t = i_online[items_t]
            i_target_t = i_target[items_t]
            score_mat_ui = torch.sum(torch.mul(u_online_t, i_target_t).squeeze(), dim=1)
            score_mat_iu = torch.sum(torch.mul(u_target_t, i_online_t).squeeze(), dim=1)
            score_mat = score_mat_ui + score_mat_iu

        return score_mat


class LGCN_Encoder(nn.Module):
    def __init__(
        self, user_count, item_count, latent_size, norm_adj, n_layers=3, drop_flag=False
    ):
        super(LGCN_Encoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_size = latent_size
        self.layers = [latent_size] * n_layers

        self.norm_adj = norm_adj

        self.drop_ratio = 0.2
        self.drop_flag = drop_flag

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = norm_adj

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict(
            {
                "user_emb": nn.Parameter(
                    initializer(torch.empty(self.user_count, self.latent_size))
                ),
                "item_emb": nn.Parameter(
                    initializer(torch.empty(self.item_count, self.latent_size))
                ),
            }
        )

        return embedding_dict

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).cuda()
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).cuda()
        return out * (1.0 / (1 - rate))

    def forward(self, users, items):
        A_hat = (
            self.sparse_dropout(
                self.sparse_norm_adj,
                np.random.random() * self.drop_ratio,
                self.sparse_norm_adj._nnz(),
            )
            if self.drop_flag
            else self.sparse_norm_adj
        )

        ego_embeddings = torch.cat(
            [self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]], 0
        )
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[: self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count :, :]

        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings

    @torch.no_grad()
    def get_embedding(self):
        A_hat = self.sparse_norm_adj

        ego_embeddings = torch.cat(
            [self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]], 0
        )
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[: self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count :, :]

        return user_all_embeddings, item_all_embeddings


class BUIREngine(ModelEngine):
    """BUIREngine Class."""

    # A class includes train an epoch and train a batch of BUIR

    def __init__(self, config):
        """Initialize BUIREngine Class."""
        self.config = config
        self.norm_adj = config["model"]["norm_adj"]
        self.model = BUIR_NB(config["model"])
        super(BUIREngine, self).__init__(config)
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

        users, pos_items, neg_items = batch_data

        output = self.model(users, pos_items)
        batch_loss = self.model.get_loss(output)
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

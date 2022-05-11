import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse

from beta_rec.models.torch_engine import ModelEngine


class SimGCL(torch.nn.Module):
    def __init__(self, config, norm_adj):
        super(SimGCL, self).__init__()
        self.config = config
        self.eps = float(config["eps"])
        self.n_layers = int(config["n_layer"])
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.norm_adj = norm_adj
        self.init_emb()

    def init_emb(self):
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_embedding.weight), dim=0
        )

        norm_adj = self.norm_adj.to(self.device)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = sparse.mm(norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings)
                ego_embeddings += (
                    torch.sign(ego_embeddings)
                    * F.normalize(random_noise, dim=-1)
                    * self.eps
                )
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def infoNCE(self, view1, view2, temperature):
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score).sum()
        return cl_loss

    def cal_cl_loss(self, user_index, item_index):
        u_idx = torch.unique(user_index)
        i_idx = torch.unique(item_index)
        user_view_1, item_view_1 = self.forward(perturbed=True)
        user_view_1, item_view_1 = F.normalize(user_view_1, dim=1), F.normalize(
            item_view_1, dim=1
        )
        user_view_2, item_view_2 = self.forward(perturbed=True)
        user_view_2, item_view_2 = F.normalize(user_view_1, dim=1), F.normalize(
            item_view_2, dim=1
        )
        user_cl_loss = self.infoNCE(user_view_1[u_idx, :], user_view_2[u_idx, :], 0.2)
        item_cl_loss = self.infoNCE(item_view_1[i_idx, :], item_view_2[i_idx, :], 0.2)
        return user_cl_loss + item_cl_loss

    def predict(self, users, items):

        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)

        with torch.no_grad():
            score = torch.mul(
                self.user_embedding.weight[users_t], self.item_embedding.weight[items_t]
            ).sum(dim=1)

        return score


class SimGCLEngine(ModelEngine):
    """SimGCLEngine Class."""

    # A class includes train an epoch and train a batch of SimGCL

    def __init__(self, config):
        """Initialize SimGCLEngine Class."""
        self.config = config
        self.reg = float(config["model"]["reg"])
        self.batch_size = config["model"]["batch_size"]
        self.cl_rate = float(config["model"]["lambda"])
        self.norm_adj = config["model"]["norm_adj"]
        self.model = SimGCL(config["model"], self.norm_adj)
        super(SimGCLEngine, self).__init__(config)
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
        rec_user_emb, rec_item_emb = self.model.forward()

        batch_users, pos_items, neg_items = batch_data

        user_emb, pos_item_emb, neg_item_emb = (
            rec_user_emb[batch_users],
            rec_item_emb[pos_items],
            rec_item_emb[neg_items],
        )

        rec_loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)

        cl_loss = self.cl_rate * self.model.cal_cl_loss(batch_users, pos_items)
        batch_loss = (
            rec_loss
            + self.l2_reg_loss(self.reg, *[user_emb, pos_item_emb, neg_item_emb])
            + cl_loss
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
        regularizer = 0.0
        for batch_data in train_loader:
            loss = self.train_single_batch(batch_data)
            total_loss += loss
        print(f"[Training Epoch {epoch_id}], Loss {loss}")
        self.writer.add_scalar("model/loss", total_loss, epoch_id)
        self.writer.add_scalar("model/regularizer", regularizer, epoch_id)

    def bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
        loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score)).sum()
        return loss

    def l2_reg_loss(self, reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)
        return emb_loss * reg

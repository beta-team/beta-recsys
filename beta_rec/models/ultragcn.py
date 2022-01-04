import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from beta_rec.models.torch_engine import ModelEngine


def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero=False):
    print("Computing \\Omega for the item-item graph... ")
    A = train_mat.T.dot(train_mat)  # I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis=0).reshape(-1)
    users_D = np.sum(A, axis=1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print("i-i constraint matrix {} ok".format(i))

    print("Computation \\Omega OK!")
    return res_mat.long(), res_sim_mat.float()


class UltraGCN(nn.Module):
    def __init__(self, config):
        super(UltraGCN, self).__init__()
        self.config = config
        self.user_num = config["n_users"]
        self.item_num = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.w1 = config["w1"]
        self.w2 = config["w2"]
        self.w3 = config["w3"]
        self.w4 = config["w4"]

        self.negative_weight = config["negative_weight"]
        self.gamma = config["gamma"]
        self.lambda_ = config["lambda"]

        self.user_embeds = nn.Embedding(self.user_num, self.emb_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.emb_dim)

        self.train_mat = config["train_mat"]
        self.constraint_mat = config["constraint_mat"]
        self.constraint_mat["beta_uD"] = torch.tensor(self.constraint_mat["beta_uD"])[0]
        self.constraint_mat["beta_iD"] = torch.tensor(self.constraint_mat["beta_iD"])[0]

        self.ii_neighbor_num = config["ii_neighbor_num"]

        ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(
            self.train_mat, self.ii_neighbor_num
        )
        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat

        self.initial_weights()

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=1e-3)
        nn.init.normal_(self.item_embeds.weight, std=1e-3)

    def get_omegas(self, users, pos_items, neg_items):
        if self.w2 > 0:
            pos_weight = torch.mul(
                self.constraint_mat["beta_uD"][users],
                self.constraint_mat["beta_iD"][pos_items],
            ).to(self.device)
            pow_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(self.device)

        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(
                torch.repeat_interleave(
                    self.constraint_mat["beta_uD"][users], neg_items.size(1)
                ),
                self.constraint_mat["beta_iD"][neg_items.flatten()],
            ).to(self.device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(
                self.device
            )

        weight = torch.cat((pow_weight, neg_weight))
        return weight

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)  # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1)  # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(self.device)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores,
            neg_labels,
            weight=omega_weight[len(pos_scores) :].view(neg_scores.size()),
            reduction="none",
        ).mean(dim=-1)

        pos_labels = torch.ones(pos_scores.size()).to(self.device)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores,
            pos_labels,
            weight=omega_weight[: len(pos_scores)],
            reduction="none",
        )

        loss = pos_loss + neg_loss * self.negative_weight

        del pos_scores, neg_scores, pos_loss, neg_loss, pos_labels, neg_labels

        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        neighbor_embeds = self.item_embeds(
            self.ii_neighbor_mat[pos_items].to(self.device)
        )  # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(
            self.device
        )  # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users).unsqueeze(1)

        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()

        # loss = loss.sum(-1)
        del neighbor_embeds, sim_scores

        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self, users, pos_items, neg_items):
        omega_weight = self.get_omegas(users, pos_items, neg_items)

        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss

    def predict(self, users, items):

        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)
        with torch.no_grad():

            user_embeds = self.user_embeds(users_t)
            item_embeds = self.item_embeds(items_t)

            scores = torch.sum(torch.mul(user_embeds, item_embeds).squeeze(), dim=1)

        return scores


class UltraGCNEngine(ModelEngine):
    """UltraGCNEngine Class."""

    # A class includes train an epoch and train a batch of UltraGCN

    def __init__(self, config):
        """Initialize UltraGCNEngine Class."""
        self.config = config
        self.regs = config["model"]["regs"]  # reg is the regularisation
        self.decay = self.regs[0]
        self.model = UltraGCN(config["model"])
        super(UltraGCNEngine, self).__init__(config)
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

        batch_users, pos_items, neg_items = batch_data

        batch_loss = self.model.forward(
            batch_users,
            pos_items,
            neg_items,
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

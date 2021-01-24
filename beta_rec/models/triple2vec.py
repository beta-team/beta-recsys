import torch
import torch.nn as nn
import torch.nn.functional as F

from beta_rec.models.torch_engine import ModelEngine


class Triple2vec(nn.Module):
    """Triple2vec Class."""

    def __init__(self, config):
        """Initialize Triple2vec Class."""
        super(Triple2vec, self).__init__()
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.n_neg = config["n_neg"]
        self.use_bias = config["n_neg"]
        self.batch_size = config["batch_size"]
        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb1 = nn.Embedding(self.n_items, self.emb_dim)
        self.item_emb2 = nn.Embedding(self.n_items, self.emb_dim)
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        self.init_emb()

    def init_emb(self):
        """Initialize embeddings."""
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_emb1.weight.data.uniform_(-0.01, 0.01)
        self.item_emb2.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.fill_(0.0)
        self.item_bias.weight.data.fill_(0.0)

    def forward(self, batch_data):
        """Train the model."""
        if self.use_bias:
            self.item_emb2 = self.item_emb1
        pos_u, pos_i_1, pos_i_2, neg_u, neg_i_1, neg_i_2 = batch_data
        emb_u = self.user_emb(pos_u)
        emb_i_1 = self.item_emb1(pos_i_1)
        emb_i_2 = self.item_emb2(pos_i_2)

        emb_u_neg = self.user_emb(neg_u)
        emb_i_1_neg = self.item_emb1(neg_i_2)
        emb_i_2_neg = self.item_emb2(neg_i_2)

        input_emb_u = emb_i_1 + emb_i_2
        u_pos_score = torch.mul(emb_u, input_emb_u).squeeze()
        u_pos_score = torch.sum(u_pos_score, dim=1) + self.user_bias(pos_u).squeeze()
        u_pos_score = F.logsigmoid(u_pos_score)

        u_neg_score = (
            torch.bmm(emb_u_neg, emb_u.unsqueeze(2)).squeeze()
            + self.user_bias(neg_u).squeeze()
        )
        u_neg_score = F.logsigmoid(-1 * u_neg_score)
        u_score = -1 * (torch.sum(u_pos_score) + torch.sum(u_neg_score))

        input_emb_i_1 = emb_u + emb_i_2
        i_1_pos_score = torch.mul(emb_i_1, input_emb_i_1).squeeze()
        i_1_pos_score = (
            torch.sum(i_1_pos_score, dim=1) + self.item_bias(pos_i_1).squeeze()
        )
        i_1_pos_score = F.logsigmoid(i_1_pos_score)

        i_1_neg_score = (
            torch.bmm(emb_i_1_neg, emb_i_1.unsqueeze(2)).squeeze()
            + self.item_bias(neg_i_1).squeeze()
        )
        i_1_neg_score = F.logsigmoid(-1 * i_1_neg_score)

        i_1_score = -1 * (torch.sum(i_1_pos_score) + torch.sum(i_1_neg_score))

        input_emb_i_2 = emb_u + emb_i_1
        i_2_pos_score = torch.mul(emb_i_2, input_emb_i_2).squeeze()
        i_2_pos_score = (
            torch.sum(i_2_pos_score, dim=1) + self.item_bias(pos_i_2).squeeze()
        )
        i_2_pos_score = F.logsigmoid(i_2_pos_score)

        i_2_neg_score = (
            torch.bmm(emb_i_2_neg, emb_i_2.unsqueeze(2)).squeeze()
            + self.item_bias(neg_i_2).squeeze()
        )
        i_2_neg_score = F.logsigmoid(-1 * i_2_neg_score)

        i_2_score = -1 * (torch.sum(i_2_pos_score) + torch.sum(i_2_neg_score))

        return (u_score + i_1_score + i_2_score) / (3 * self.batch_size)

    def predict(self, users, items):
        """Predict result with the model."""
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            scores = torch.mul(
                self.user_emb(users_t),
                (self.item_emb1(items_t) + self.item_emb2(items_t)) / 2,
            ).sum(dim=1)
        return scores


class Triple2vecEngine(ModelEngine):
    """Engine for training Triple model."""

    def __init__(self, config):
        """Initialize Triple2vecEngine Class."""
        self.config = config
        self.model = Triple2vec(config["model"])
        super(Triple2vecEngine, self).__init__(config)

    def train_single_batch(self, batch_data, ratings=None):
        """Train the model in a single batch."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        loss = self.model.forward(batch_data)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        """Train the model in one epoch."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        for batch_id, sample in enumerate(train_loader):
            pos_u = torch.tensor(
                [triple[0] for triple in sample],
                dtype=torch.int64,
                device=self.device,
            )
            pos_i_1 = torch.tensor(
                [triple[1] for triple in sample],
                dtype=torch.int64,
                device=self.device,
            )
            pos_i_2 = torch.tensor(
                [triple[2] for triple in sample],
                dtype=torch.int64,
                device=self.device,
            )
            neg_u = torch.tensor(
                self.data.user_sampler.sample(
                    self.config["model"]["n_neg"], len(sample)
                ),
                dtype=torch.int64,
                device=self.device,
            )
            neg_i_1 = torch.tensor(
                self.data.item_sampler.sample(
                    self.config["model"]["n_neg"], len(sample)
                ),
                dtype=torch.int64,
                device=self.device,
            )

            neg_i_2 = torch.tensor(
                self.data.item_sampler.sample(
                    self.config["model"]["n_neg"], len(sample)
                ),
                dtype=torch.int64,
                device=self.device,
            )
            batch_data = (pos_u, pos_i_1, pos_i_2, neg_u, neg_i_1, neg_i_2)
            loss = self.train_single_batch(batch_data)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

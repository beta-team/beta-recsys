import torch
import torch.nn as nn
import torch.nn.functional as F

from beta_rec.models.torch_engine import ModelEngine
from beta_rec.utils.common_util import timeit


class VBCAR(nn.Module):
    """VBCAR Class."""

    def __init__(self, config):
        """Initialize VBCAR Class."""
        super(VBCAR, self).__init__()
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.late_dim = config["late_dim"]
        self.emb_dim = config["emb_dim"]
        self.n_neg = config["n_neg"]
        self.batch_size = config["batch_size"]
        self.alpha = config["alpha"]
        self.esp = 1e-10
        if config["activator"] == "tanh":
            self.act = torch.tanh
        elif config["activator"] == "sigmoid":
            self.act = torch.sigmoid
        elif config["activator"] == "relu":
            self.act = F.relu
        elif config["activator"] == "lrelu":
            self.act = F.leaky_relu
        elif config["activator"] == "prelu":
            self.act = F.prelu
        else:
            self.act = lambda x: x

    def init_layers(self):
        """Initialize layers in the model."""
        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        init_range = 0.1 * (self.emb_dim) ** (-1 / 2)
        self.item_emb.weight.data.uniform_(-init_range, init_range)
        self.fc_u_1_mu = nn.Linear(self.user_fea_dim, self.late_dim)
        self.fc_u_2_mu = nn.Linear(self.late_dim, self.emb_dim * 2)
        self.fc_i_1_mu = nn.Linear(self.item_fea_dim, self.late_dim)
        self.fc_i_2_mu = nn.Linear(self.late_dim, self.emb_dim * 2)

    def init_feature(self, user_fea, item_fea):
        """Initialize features."""
        self.user_fea = user_fea
        self.item_fea = item_fea
        self.user_fea_dim = user_fea.size()[1]
        self.item_fea_dim = item_fea.size()[1]

    def user_encode(self, index):
        """Encode user."""
        x = self.user_fea[index]
        x = self.fc_u_2_mu(self.act(self.fc_u_1_mu(x)))
        mu = x[:, : self.emb_dim]
        std = x[:, self.emb_dim :]
        return mu, std

    def item_encode(self, index):
        """Encode item."""
        x = self.item_fea[index]
        x = self.fc_i_2_mu(self.act(self.fc_i_1_mu(x)))
        mu = x[:, : self.emb_dim]
        std = x[:, self.emb_dim :]
        return mu, std

    def reparameterize(self, gaussian):
        """Re-parameterize the model."""
        mu, std = gaussian
        std = torch.exp(0.5 * std)
        eps = torch.randn_like(std)
        return mu + std * eps

    def kl_div(self, dis1, dis2=None, neg=False):
        """Missing Doc."""
        mean1, std1 = dis1
        if dis2 is None:
            mean2 = torch.zeros(mean1.size(), device=self.device)
            std2 = torch.ones(mean1.size(), device=self.device)
        else:
            mean2, std2 = dis2
        var1 = std1.pow(2) + self.esp
        var2 = std2.pow(2) + self.esp
        mean_pow2 = (
            (mean2 - mean1)
            * (torch.tensor(1.0, device=self.device) / var2)
            * (mean2 - mean1)
        )
        tr_std_mul = (torch.tensor(1.0, device=self.device) / var2) * var1
        if neg is False:
            dkl = (
                (torch.log(var2 / var1) - 1 + tr_std_mul + mean_pow2)
                .mul(0.5)
                .sum(dim=1)
                .mean()
            )
        else:
            dkl = (
                (torch.log(var2 / var1) - 1 + tr_std_mul + mean_pow2)
                .mul(0.5)
                .sum(dim=2)
                .sum(dim=1)
                .mean()
            )
        return dkl

    def forward(self, batch_data):
        """Train the model."""
        pos_u, pos_i_1, pos_i_2, neg_u, neg_i_1, neg_i_2 = batch_data
        pos_u_dis = self.user_encode(pos_u)
        emb_u = torch.cat((self.reparameterize(pos_u_dis), self.user_emb(pos_u)), dim=1)
        pos_i_1_dis = self.item_encode(pos_i_1)
        emb_i_1 = torch.cat(
            (self.reparameterize(pos_i_1_dis), self.item_emb(pos_i_1)), dim=1
        )
        pos_i_2_dis = self.item_encode(pos_i_2)
        emb_i_2 = torch.cat(
            (self.reparameterize(pos_i_2_dis), self.item_emb(pos_i_2)), dim=1
        )
        neg_u_dis = self.user_encode(neg_u.view(-1))
        emb_u_neg = torch.cat(
            (self.reparameterize(neg_u_dis), self.user_emb(neg_u.view(-1))), dim=1
        ).view(-1, self.n_neg, self.emb_dim * 2)
        neg_i_1_dis = self.item_encode(neg_i_1.view(-1))
        emb_i_1_neg = torch.cat(
            (self.reparameterize(neg_i_1_dis), self.item_emb(neg_i_1.view(-1))), dim=1
        ).view(-1, self.n_neg, self.emb_dim * 2)
        neg_i_2_dis = self.item_encode(neg_i_2.view(-1))
        emb_i_2_neg = torch.cat(
            (self.reparameterize(neg_i_2_dis), self.item_emb(neg_i_2.view(-1))), dim=1
        ).view(-1, self.n_neg, self.emb_dim * 2)

        input_emb_u = emb_i_1 + emb_i_2
        u_pos_score = torch.mul(emb_u, input_emb_u).squeeze()
        u_pos_score = torch.sum(u_pos_score, dim=1)
        u_pos_score = F.logsigmoid(u_pos_score)

        u_neg_score = torch.bmm(emb_u_neg, emb_u.unsqueeze(2)).squeeze()
        u_neg_score = F.logsigmoid(-1 * u_neg_score)
        u_score = -1 * (torch.sum(u_pos_score) + torch.sum(u_neg_score))

        input_emb_i_1 = emb_u + emb_i_2
        i_1_pos_score = torch.mul(emb_i_1, input_emb_i_1).squeeze()
        i_1_pos_score = torch.sum(i_1_pos_score, dim=1)
        i_1_pos_score = F.logsigmoid(i_1_pos_score)

        i_1_neg_score = torch.bmm(emb_i_1_neg, emb_i_1.unsqueeze(2)).squeeze()
        i_1_neg_score = F.logsigmoid(-1 * i_1_neg_score)

        i_1_score = -1 * (torch.sum(i_1_pos_score) + torch.sum(i_1_neg_score))

        input_emb_i_2 = emb_u + emb_i_1
        i_2_pos_score = torch.mul(emb_i_2, input_emb_i_2).squeeze()
        i_2_pos_score = torch.sum(i_2_pos_score, dim=1)
        i_2_pos_score = F.logsigmoid(i_2_pos_score)

        i_2_neg_score = torch.bmm(emb_i_2_neg, emb_i_2.unsqueeze(2)).squeeze()
        i_2_neg_score = F.logsigmoid(-1 * i_2_neg_score)

        i_2_score = -1 * (torch.sum(i_2_pos_score) + torch.sum(i_2_neg_score))

        GEN = (u_score + i_1_score + i_2_score) / (3 * self.batch_size)
        KLD = (
            self.kl_div(pos_u_dis)
            + self.kl_div(pos_i_1_dis)
            + self.kl_div(pos_i_2_dis)
            + self.kl_div(neg_u_dis)
            + self.kl_div(neg_i_1_dis)
            + self.kl_div(neg_i_2_dis)
        ) / (3 * self.batch_size)
        self.kl_loss = KLD.item()
        self.rec_loss = GEN.item()
        # return GEN
        return (1 - self.alpha) * (GEN) + (self.alpha * KLD)

    def predict(self, users, items):
        """Predict result with the model."""
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            scores = torch.mul(
                torch.cat(
                    (self.user_encode(users_t)[0], self.user_emb(users_t)), dim=1
                ),
                torch.cat(
                    (self.item_encode(items_t)[0], self.item_emb(items_t)), dim=1
                ),
            ).sum(dim=1)
        return scores


class VBCAREngine(ModelEngine):
    """Engine for training & evaluating GMF model."""

    def __init__(self, config):
        """Initialize VBCAREngine Class."""
        self.config = config
        self.model = VBCAR(config["model"])
        user_fea = torch.tensor(
            config["user_fea"],
            requires_grad=False,
            device=config["model"]["device_str"],
            dtype=torch.float32,
        )
        item_fea = torch.tensor(
            config["item_fea"],
            requires_grad=False,
            device=config["model"]["device_str"],
            dtype=torch.float32,
        )
        self.model.init_feature(user_fea, item_fea)
        self.model.init_layers()
        super(VBCAREngine, self).__init__(config)

    def train_single_batch(self, batch_data, ratings=None):
        """Train the model in a single batch."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        loss = self.model.forward(batch_data)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    @timeit
    def train_an_epoch(self, train_loader, epoch_id):
        """Train the model in one epoch."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        kl_loss = 0
        rec_loss = 0
        #         with autograd.detect_anomaly():
        for batch_id, sample in enumerate(train_loader):
            assert isinstance(sample, torch.Tensor)
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
            kl_loss += self.model.kl_loss
            rec_loss += self.model.rec_loss
        total_loss = total_loss / self.config["model"]["batch_size"]
        rec_loss = rec_loss / self.config["model"]["batch_size"]
        kl_loss = kl_loss / self.config["model"]["batch_size"]
        print(
            "[Training Epoch {}], log_like_loss {} kl_loss: {} alpha: {} lr: {}".format(
                epoch_id,
                rec_loss,
                kl_loss,
                self.model.alpha,
                self.config["model"]["lr"],
            )
        )
        self.writer.add_scalars(
            "model/loss",
            {
                "total_loss": total_loss,
                "rec_loss": total_loss - kl_loss,
                "kl_loss": kl_loss,
            },
            epoch_id,
        )

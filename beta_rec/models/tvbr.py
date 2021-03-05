import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from beta_rec.models.torch_engine import ModelEngine
from beta_rec.utils.common_util import timeit


class TVBR(nn.Module):
    """TVBR pytorch Module.

    Args:
        nn ([type]): [description]
    """

    def __init__(self, config):
        """Initialize the  pytorch Module.

        Args:
            config ([type]): [description]
        """
        super(TVBR, self).__init__()
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.late_dim = config["late_dim"]
        self.emb_dim = config["emb_dim"]
        self.n_neg = config["n_neg"]
        self.batch_size = config["batch_size"]
        self.alpha = config["alpha"]
        self.time_step = config["time_step"]
        self.esp = 1e-10
        if config["activator"] == "tanh":
            self.act = nn.Tanh
        elif config["activator"] == "hardtanh":
            self.act = nn.Hardtanh
        elif config["activator"] == "logsigmoid":
            self.act = nn.LogSigmoid
        elif config["activator"] == "sigmoid":
            self.act = nn.Sigmoid
        elif config["activator"] == "relu":
            self.act = nn.ReLU
        elif config["activator"] == "lrelu":
            self.act = nn.LeakyReLU
        elif config["activator"] == "prelu":
            self.act = nn.PReLU
        else:
            self.act = lambda x: x
        self.esp = 1e-10

    def init_feature(self, user_fea, item_fea):
        """Initialize features of users and items.

        Args:
            user_fea ([type]): [description]
            item_fea ([type]): [description]
        """
        self.user_fea = user_fea
        self.item_fea = item_fea
        self.user_fea_dim = user_fea.size()[1]
        self.item_fea_dim = item_fea.size()[1]

    def init_layers(self):
        """Initialize Model Layers."""
        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        init_range = 0.1 * (self.emb_dim ** (-1 / 2))
        self.user_emb.weight.data.uniform_(-init_range, init_range)
        self.item_emb.weight.data.uniform_(-init_range, init_range)

        self.time_embdding = nn.Embedding(
            self.time_step + 1,
            self.time_step + 1,
            _weight=torch.eye(self.time_step + 1, self.time_step + 1),
        )

        self.user_mean = nn.Embedding(
            self.n_users, self.emb_dim, _weight=torch.ones(self.n_users, self.emb_dim)
        )
        self.user_std = nn.Embedding(
            self.n_users, self.emb_dim, _weight=torch.zeros(self.n_users, self.emb_dim)
        )

        self.item_mean = nn.Embedding(
            self.n_items, self.emb_dim, _weight=torch.ones(self.n_items, self.emb_dim)
        )
        self.item_std = nn.Embedding(
            self.n_items, self.emb_dim, _weight=torch.zeros(self.n_items, self.emb_dim)
        )

        self.item_mean.weight.data.uniform_(-init_range, init_range)
        self.item_std.weight.data.uniform_(-init_range, init_range)

        self.time2mean_u = nn.Sequential(
            nn.Linear(
                self.emb_dim + self.time_step + 1 + self.user_fea_dim, self.emb_dim
            ),
            self.act(),
        )
        self.time2std_u = nn.Sequential(
            nn.Linear(
                self.emb_dim + self.time_step + 1 + self.user_fea_dim, self.emb_dim
            ),
            self.act(),
        )
        self.time2mean_i = nn.Sequential(
            nn.Linear(
                self.emb_dim + self.time_step + 1 + self.item_fea_dim, self.emb_dim
            ),
            self.act(),
        )
        self.time2std_i = nn.Sequential(
            nn.Linear(
                self.emb_dim + self.time_step + 1 + self.item_fea_dim, self.emb_dim
            ),
            self.act(),
        )

    def user_encode(self, index, time_laten, pri_time_laten):
        """Encode user embedding.

        Args:
            index ([type]): [description]
            time_laten ([type]): [description]
            pri_time_laten ([type]): [description]

        Returns:
            [type]: [description]
        """
        user_mean = self.user_mean(index).squeeze(1)
        user_mean_pri = self.time2mean_u(
            torch.cat([user_mean, pri_time_laten, self.user_fea[index]], 1)
        )
        user_mean = self.time2mean_u(
            torch.cat([user_mean, time_laten, self.user_fea[index]], 1)
        )

        user_std = self.user_std(index).squeeze(1)
        user_std_pri = (
            self.time2std_u(
                torch.cat([user_std, pri_time_laten, self.user_fea[index]], 1)
            )
            .mul(0.5)
            .exp()
        )
        user_std = (
            self.time2std_u(torch.cat([user_std, time_laten, self.user_fea[index]], 1))
            .mul(0.5)
            .exp()
        )
        return ((user_mean_pri, user_std_pri), (user_mean, user_std))

    def item_encode(self, index, time_laten, pri_time_laten):
        """Encode item embedding.

        Args:
            index ([type]): [description]
            time_laten ([type]): [description]
            pri_time_laten ([type]): [description]

        Returns:
            [type]: [description]
        """
        item_mean = self.item_mean(index).squeeze(1)
        item_mean_pri = self.time2mean_i(
            torch.cat([item_mean, pri_time_laten, self.item_fea[index]], 1)
        )
        item_mean = self.time2mean_i(
            torch.cat([item_mean, time_laten, self.item_fea[index]], 1)
        )

        item_std = self.item_std(index).squeeze(1)
        item_std_pri = (
            self.time2std_i(
                torch.cat([item_std, pri_time_laten, self.item_fea[index]], 1)
            )
            .mul(0.5)
            .exp()
        )
        item_std = (
            self.time2std_i(torch.cat([item_std, time_laten, self.item_fea[index]], 1))
            .mul(0.5)
            .exp()
        )
        return ((item_mean_pri, item_std_pri), (item_mean, item_std))

    def reparameterize(self, mu, logvar):
        """Reparameterize the Gaussian Distribution for back propagation.

        Args:
            mu ([type]): [description]
            logvar ([type]): [description]

        Returns:
            [type]: [description]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_div(self, dis1, dis2, neg=False):
        """Compute the KL divergence for two Gaussian distributions.

        Args:
            dis1 (tuple<mean,std>): Gaussian distribution 1
            dis2 (tuple<mean,std>), optional): Gaussian distribution 2. Defaults to None. If None, a
            standard normal distribution will be applied
            neg (bool, optional): If contains negative samples. Defaults to False.

        Returns:
            [type]: [description]
        """
        mean1, std1 = dis1
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
        """Forward batch_data.

        Args:
            batch_data (tuple): Input batch data.

        Returns:
            [type]: [description]
        """
        (
            pos_u,
            pos_i_1,
            pos_i_2,
            neg_u,
            neg_i_1,
            neg_i_2,
            pos_batch_t,
            neg_batch_t,
        ) = batch_data

        """
        time embedding
        """
        pos_time_laten = self.time_embdding(
            pos_batch_t + torch.tensor(1).to(self.device)
        ).squeeze(1)
        pos_pri_time_laten = self.time_embdding(pos_batch_t)

        neg_time_laten = self.time_embdding(
            neg_batch_t + torch.tensor(1).to(self.device)
        ).squeeze(1)
        neg_pri_time_laten = self.time_embdding(neg_batch_t)

        """
        positive user embeddings
        """
        pos_u_dis_pri, pos_u_dis = self.user_encode(
            pos_u, pos_time_laten, pos_pri_time_laten
        )
        pos_u_emb = self.reparameterize(pos_u_dis[0], pos_u_dis[1])
        pos_u_kl = self.kl_div(pos_u_dis_pri, pos_u_dis, False)

        """
        positive item embeddings
        """
        pos_i_1_dis_pri, pos_i_1_dis = self.item_encode(
            pos_i_1, pos_time_laten, pos_pri_time_laten
        )
        pos_i_1_emb = self.reparameterize(pos_i_1_dis[0], pos_i_1_dis[1])
        pos_i_1_kl = self.kl_div(pos_i_1_dis_pri, pos_i_1_dis, False)

        pos_i_2_dis_pri, pos_i_2_dis = self.item_encode(
            pos_i_2, pos_time_laten, pos_pri_time_laten
        )
        pos_i_2_emb = self.reparameterize(pos_i_2_dis[0], pos_i_2_dis[1])
        pos_i_2_kl = self.kl_div(pos_i_2_dis_pri, pos_i_2_dis, False)

        """
        negative user embeddings
        """
        neg_u_dis_pri, neg_u_dis = self.user_encode(
            neg_u.view(-1), neg_time_laten, neg_pri_time_laten
        )
        neg_u_emb = self.reparameterize(neg_u_dis[0], neg_u_dis[1])
        neg_u_kl = self.kl_div(neg_u_dis_pri, neg_u_dis, False)

        """
        negative item embeddings
        """
        neg_i_1_dis_pri, neg_i_1_dis = self.item_encode(
            neg_i_1.view(-1), neg_time_laten, neg_pri_time_laten
        )
        neg_i_1_emb = self.reparameterize(neg_i_1_dis[0], neg_i_1_dis[1])
        neg_i_1_kl = self.kl_div(neg_i_1_dis_pri, neg_i_1_dis, False)

        neg_i_2_dis_pri, neg_i_2_dis = self.item_encode(
            neg_i_2.view(-1), neg_time_laten, neg_pri_time_laten
        )
        neg_i_2_emb = self.reparameterize(neg_i_2_dis[0], neg_i_2_dis[1])
        neg_i_2_kl = self.kl_div(neg_i_2_dis_pri, neg_i_2_dis, False)

        pos_u_emb = torch.cat((pos_u_emb, self.user_emb(pos_u)), dim=1)
        pos_i_1_emb = torch.cat((pos_i_1_emb, self.item_emb(pos_i_1)), dim=1)
        pos_i_2_emb = torch.cat((pos_i_2_emb, self.item_emb(pos_i_2)), dim=1)
        neg_u_emb = torch.cat((neg_u_emb, self.user_emb(neg_u.view(-1))), dim=1).view(
            -1, self.n_neg, self.emb_dim * 2
        )

        neg_i_1_emb = torch.cat(
            (neg_i_1_emb, self.item_emb(neg_i_1.view(-1))), dim=1
        ).view(-1, self.n_neg, self.emb_dim * 2)

        neg_i_2_emb = torch.cat(
            (neg_i_2_emb, self.item_emb(neg_i_2.view(-1))), dim=1
        ).view(-1, self.n_neg, self.emb_dim * 2)

        input_emb_u = pos_i_1_emb + pos_i_2_emb
        u_pos_score = torch.mul(pos_u_emb, input_emb_u).squeeze()
        u_pos_score = torch.sum(u_pos_score, dim=1)
        u_pos_score = F.logsigmoid(u_pos_score)

        u_neg_score = torch.bmm(neg_u_emb, pos_u_emb.unsqueeze(2)).squeeze()
        u_neg_score = F.logsigmoid(-1 * u_neg_score)
        u_score = -1 * (torch.sum(u_pos_score) + torch.sum(u_neg_score))

        input_emb_i_1 = pos_u_emb + pos_i_2_emb
        i_1_pos_score = torch.mul(pos_i_1_emb, input_emb_i_1).squeeze()
        i_1_pos_score = torch.sum(i_1_pos_score, dim=1)
        i_1_pos_score = F.logsigmoid(i_1_pos_score)
        i_1_neg_score = torch.bmm(neg_i_1_emb, pos_i_1_emb.unsqueeze(2)).squeeze()
        i_1_neg_score = F.logsigmoid(-1 * i_1_neg_score)
        i_1_score = -1 * (torch.sum(i_1_pos_score) + torch.sum(i_1_neg_score))

        input_emb_i_2 = pos_u_emb + pos_i_1_emb
        i_2_pos_score = torch.mul(pos_i_2_emb, input_emb_i_2).squeeze()
        i_2_pos_score = torch.sum(i_2_pos_score, dim=1)
        i_2_pos_score = F.logsigmoid(i_2_pos_score)

        i_2_neg_score = torch.bmm(neg_i_2_emb, pos_i_2_emb.unsqueeze(2)).squeeze()
        i_2_neg_score = F.logsigmoid(-1 * i_2_neg_score)
        i_2_score = -1 * (torch.sum(i_2_pos_score) + torch.sum(i_2_neg_score))

        self.rec_loss = (u_score + i_1_score + i_2_score) / (3 * self.batch_size)
        self.kl_loss = -0.5 * (
            pos_u_kl + pos_i_1_kl + pos_i_2_kl + neg_u_kl + neg_i_1_kl + neg_i_2_kl
        )

        return (1 - self.alpha) * self.rec_loss + (self.alpha * self.kl_loss)

    def predict(self, users, items):
        """Prediction scres for user item pairs.

        Args:
            config ([type]): [description]
        """
        with torch.no_grad():
            users = torch.tensor(users, dtype=torch.int64, device=self.device)
            items = torch.tensor(items, dtype=torch.int64, device=self.device)
            times = torch.tensor(
                [self.time_step] * len(users), dtype=torch.int64, device=self.device
            )
            """
            time embedding
            """
            time_laten = self.time_embdding(times).squeeze(1)
            pri_time_laten = self.time_embdding(times - 1)

            """
            positive user embeddings
            """
            pos_u_dis_pri, pos_u_dis = self.user_encode(
                users, time_laten, pri_time_laten
            )

            """
            positive item embeddings
            """
            pos_i_dis_pri, pos_i_dis = self.item_encode(
                items, time_laten, pri_time_laten
            )
            u_embeddings = torch.cat((pos_u_dis[0], self.user_emb(users)), dim=1).view(
                len(users), 1, self.emb_dim * 2
            )
            i_embeddings = torch.cat((pos_i_dis[0], self.item_emb(items)), dim=1).view(
                len(items), self.emb_dim * 2, 1
            )
            scores = torch.bmm(
                u_embeddings,
                i_embeddings,
            ).squeeze()
            return scores


class TVBREngine(ModelEngine):
    """Model Engine For TVBR.

    Args:
        ModelEngine ([type]): [description]
    """

    def __init__(self, config):
        """Initialize configs.

        Args:
            config ([type]): [description]
        """
        self.config = config
        self.time_step = config["model"]["time_step"]
        self.batch_size = config["model"]["batch_size"]
        self.n_neg = config["model"]["n_neg"]
        self.model = TVBR(config["model"])
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
        super(TVBREngine, self).__init__(config)

    def train_single_batch(self, batch_data, ratings=None):
        """Train a single batch.

        Args:
            batch_data ([type]): [description]
            ratings ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        loss = self.model.forward(batch_data)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    @timeit
    def train_an_epoch(self, triple_df, epoch_id):
        """Train an epoch.

        Args:
            triple_df ([type]): [description]
            epoch_id ([type]): [description]
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        kl_loss = 0
        rec_loss = 0
        for t in range(self.time_step):
            t_triple_df = triple_df[triple_df["T"] == t]
            train_loader = DataLoader(
                torch.tensor(t_triple_df.to_numpy()),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )
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
                pos_batch_t = torch.tensor(
                    np.ones(self.batch_size) * t,
                    dtype=torch.int64,
                    device=self.device,
                )

                neg_batch_t = torch.tensor(
                    np.ones(self.batch_size * self.n_neg) * t,
                    dtype=torch.int64,
                    device=self.device,
                )
                batch_data = (
                    pos_u,
                    pos_i_1,
                    pos_i_2,
                    neg_u,
                    neg_i_1,
                    neg_i_2,
                    pos_batch_t,
                    neg_batch_t,
                )
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

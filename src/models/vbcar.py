"""
Created on Aug 5, 2019
Update on XX,2019 BY xxx@

Classes describing datasets of user-item interactions. Instances of these
are returned by dataset fetching and dataset pre-processing functions.

@author: Zaiqiao Meng (zaiqiao.meng@gmail.com)

"""

import sys
sys.path.append("../")
from models.torch_engine import Engine
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VBCAR(nn.Module):
    def __init__(self, config):
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
            self.act = F.sigmoid
        elif config["activator"] == "relu":
            self.act = F.relu
        elif config["activator"] == "lrelu":
            self.act = F.leaky_relu
        elif config["activator"] == "prelu":
            self.act = F.prelu
        else:
            self.act = lambda x: x

    def init_layers(self):
        self.fc_u_1_mu = nn.Linear(self.user_fea_dim, self.emb_dim)
        self.fc_i_1_mu = nn.Linear(self.item_fea_dim, self.emb_dim)
        self.fc_i_2_mu = nn.Linear(self.item_fea_dim, self.emb_dim)
        self.fc_u_1_std = nn.Linear(self.user_fea_dim, self.emb_dim)
        self.fc_i_1_std = nn.Linear(self.item_fea_dim, self.emb_dim)
        self.fc_i_2_std = nn.Linear(self.item_fea_dim, self.emb_dim)

    def init_feature(self, user_fea, item_fea):
        self.user_fea = user_fea
        self.item_fea = item_fea
        self.user_fea_dim = user_fea.size()[1]
        self.item_fea_dim = item_fea.size()[1]

    def user_encode(self, index):
        x = self.user_fea[index]
        mu = torch.tanh(self.fc_u_1_mu(x))
        std = self.act(self.fc_u_1_std(x))
        return mu, std

    def item_1_encode(self, index):
        x = self.item_fea[index]
        mu = torch.tanh(self.fc_i_1_mu(x))
        std = self.act(self.fc_i_1_std(x))
        return mu, std

    def item_2_encode(self, index):
        x = self.item_fea[index]
        mu = torch.tanh(self.fc_i_2_mu(x))
        std = self.act(self.fc_i_2_std(x))
        return mu, std

    def reparameterize(self, gaussian):
        mu, std = gaussian
        std = torch.exp(0.5 * std)
        eps = torch.randn_like(std)
        return mu + eps * std

    """
    D_KL
    """

    def kl_div(self, dis1, dis2=None, neg=False):
        mean1, std1 = dis1
        if dis2 == None:
            mean2 = torch.zeros(mean1.size(), device=self.device)
            std2 = torch.ones(std1.size(), device=self.device)
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
        if neg == False:
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
        pos_u, pos_i_1, pos_i_2, neg_u, neg_i_1, neg_i_2 = batch_data
        pos_u_dis = self.user_encode(pos_u)
        emb_u = self.reparameterize(pos_u_dis)
        pos_i_1_dis = self.item_1_encode(pos_i_1)
        emb_i_1 = self.reparameterize(pos_i_1_dis)
        pos_i_2_dis = self.item_2_encode(pos_i_2)
        emb_i_2 = self.reparameterize(pos_i_2_dis)
        neg_u_dis = self.user_encode(neg_u.view(-1))
        emb_u_neg = self.reparameterize(neg_u_dis).view(-1, self.n_neg, self.emb_dim)
        neg_i_1_dis = self.item_1_encode(neg_i_1.view(-1))
        emb_i_1_neg = self.reparameterize(neg_i_1_dis).view(
            -1, self.n_neg, self.emb_dim
        )
        neg_i_2_dis = self.item_2_encode(neg_i_2.view(-1))
        emb_i_2_neg = self.reparameterize(neg_i_2_dis).view(
            -1, self.n_neg, self.emb_dim
        )

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

        GEN = (u_score + i_1_score + i_2_score) / (self.batch_size)
        KLD = (
            self.kl_div(pos_u_dis)
            + self.kl_div(pos_i_1_dis)
            + self.kl_div(pos_i_2_dis)
            + self.kl_div(neg_u_dis)
            + self.kl_div(neg_i_1_dis)
            + self.kl_div(neg_i_2_dis)
        ) / (self.batch_size)
        self.kl_loss = KLD.item()
        #         return GEN/ (3 * self.batch_size)
        return (1 - self.alpha) * (GEN) + (self.alpha * KLD)

    def predict(self, users, items):
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = to 2 rch.tensor(items, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            scores = torch.mul(
                self.user_encode(users_t)[0],
                (self.item_1_encode(items_t)[0] + self.item_2_encode(items_t)[0]) / 2,
            ).sum(dim=1)
        return scores


class VBCAREngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.config = config
        self.model = VBCAR(config)
        if config["feature_type"] == "random":
            user_fea = torch.randn(
                config["n_users"],
                self.config["late_dim"],
                dtype=torch.float32,
                device=torch.device(self.config["device_str"]),
            )
            item_fea = torch.randn(
                config["n_items"],
                self.config["late_dim"],
                dtype=torch.float32,
                device=torch.device(self.config["device_str"]),
            )
        else:
            pass
            # todo
            # user_fea, item_fea load feature
        self.model.init_feature(user_fea, item_fea)
        self.model.init_layers()
        super(VBCAREngine, self).__init__(config)

    def train_single_batch(self, batch_data, ratings=None):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        loss = self.model.forward(batch_data)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        kl_loss = 0
        #         with autograd.detect_anomaly():
        for batch_id, sample in enumerate(train_loader):
            assert isinstance(sample, torch.LongTensor)
            pos_u = torch.tensor(
                [triple[0] for triple in sample], dtype=torch.int64, device=self.device,
            )
            pos_i_1 = torch.tensor(
                [triple[1] for triple in sample], dtype=torch.int64, device=self.device,
            )
            pos_i_2 = torch.tensor(
                [triple[2] for triple in sample], dtype=torch.int64, device=self.device,
            )
            neg_u = torch.tensor(
                self.data.user_sampler.sample(self.config["n_neg"], len(sample)),
                dtype=torch.int64,
                device=self.device,
            )
            neg_i_1 = torch.tensor(
                self.data.item_sampler.sample(self.config["n_neg"], len(sample)),
                dtype=torch.int64,
                device=self.device,
            )
            neg_i_2 = torch.tensor(
                self.data.item_sampler.sample(self.config["n_neg"], len(sample)),
                dtype=torch.int64,
                device=self.device,
            )
            batch_data = (pos_u, pos_i_1, pos_i_2, neg_u, neg_i_1, neg_i_2)
            loss = self.train_single_batch(batch_data)
            total_loss += loss
            kl_loss += self.model.kl_loss
        total_loss = total_loss / self.config["batch_size"]
        kl_loss = kl_loss / self.config["batch_size"]
        print(
            "[Training Epoch {}], log_like_loss {} kl_loss: {} alpha: {} lr: {}".format(
                epoch_id,
                total_loss - kl_loss,
                kl_loss,
                self.model.alpha,
                self.config["lr"],
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
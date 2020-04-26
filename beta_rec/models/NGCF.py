import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np
from beta_rec.models.torch_engine import Engine
from beta_rec.datasets.NGCF_data_utils import Data


class NGCF(torch.nn.Module):
    def __init__(self, config):
        super(NGCF, self).__init__()
        self.config = config
        self.num_users = config["num_users"]
        self.num_items = config["num_items"]
        self.emb_dim = config["emb_dim"]
        self.layer_size = config['layer_size']
        self.n_layers = len(self.layer_size)
        self.dropout = nn.ModuleList()
        self.GC_weights = nn.ModuleList()
        self.Bi_weights = nn.ModuleList()
        self.dropout_list = list(config['mess_dropout'])
        self.layer_size = [self.emb_dim] + self.layer_size

        for i in range(self.n_layers):
            self.GC_weights.append(nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
            self.Bi_weights.append(nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
            self.dropout.append(nn.Dropout(self.dropout_list[i]))

        self.user_embedding = nn.Embedding(self.num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.emb_dim)
        self.init_emb()

    def init_emb(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, norm_adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_layers):
            side_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_weights[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_weights[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        return u_g_embeddings, i_g_embeddings

    def predict(self, users, items):
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)

        with torch.no_grad():
            scores = torch.mul(self.user_embedding(users_t),self.item_embedding(items_t)).sum(dim=1)
        return scores


class NGCFEngine(Engine):
    def __init__(self, config):
        self.config = config
        self.model = NGCF(config)
        """
        regs is regularisation
        """
        self.regs = config["regs"]
        self.decay = self.regs[0]
        self.batch_size = config["batch_size"]
        self.norm_adj = config["norm_adj"]
        self.num_batch = config["num_batch"]

        # self.data_loader = Data.__init__(path=config["path"],batch_size=config["batch_size"])
        # self.plain_adj, self.norm_adj, self.mean_adj = self.data_loader.get_adj_mat()

        super(NGCFEngine, self).__init__(config)

    def train_single_batch(self, batch_data):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        norm_adj = self.norm_adj
        ua_embeddings, ia_embeddings = self.model.forward(norm_adj)

        batch_users, pos_items, neg_items = batch_data

        u_g_embeddings = ua_embeddings[batch_users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                              neg_i_g_embeddings)

        batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

        batch_loss.backward()
        self.optimizer.step()
        loss = batch_loss.item()
        return loss

    def train_an_epoch(self, epoch_id,user,pos_i,neg_i):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0.0
        batch_data = (user, pos_i, neg_i)

        n_batch = self.num_batch

        for idx in range(n_batch):
            loss = self.train_single_batch(batch_data)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()

        # regularizer = np.sum([1./2*(ele**2).sum() for ele in (users, pos_items, neg_items)])

        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss
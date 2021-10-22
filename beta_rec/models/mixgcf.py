import torch
import torch.nn as nn

from beta_rec.models.torch_engine import ModelEngine


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(
        self,
        n_hops,
        n_users,
        interact_mat,
        device,
        edge_dropout_rate=0.5,
        mess_dropout_rate=0.1,
    ):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.device = device
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1.0 / (1 - rate))

    def forward(self, user_embed, item_embed, mess_dropout=True, edge_dropout=True):
        """A forward pass of the GCN"""
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = (
                self._sparse_dropout(self.interact_mat, self.edge_dropout_rate)
                if edge_dropout
                else self.interact_mat
            )

            agg_embed = torch.sparse.mm(interact_mat.to(self.device), agg_embed.to(self.device))
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[: self.n_users, :], embs[self.n_users :, :]


class LightGCN(torch.nn.Module):
    """Model initialisation, embedding generation and prediction of MixGCF."""

    def __init__(self, config, norm_adj):
        """Initialize LightGCN Class."""
        super(LightGCN, self).__init__()
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.pool = config["pool"]
        self.norm_adj = norm_adj
        # self.layer_size = [self.emb_dim] + self.layer_size
        self.device = config["device_str"]
        self.context_hops = config["context_hops"]
        self.mess_dropout = config["mess_dropout"]
        self.mess_dropout_rate = config["mess_dropout_rate"]
        self.edge_dropout = config["edge_dropout"]
        self.edge_dropout_rate = config["edge_dropout_rate"]

        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        self.init_emb()
        self.gcn = self.init_model()

    def init_emb(self):
        """Initialize users and items' embeddings."""
        # Initialize users and items' embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def init_model(self):
        """Initialise the GCN model"""
        return GraphConv(
            n_hops=self.context_hops,
            n_users=self.n_users,
            interact_mat=self.norm_adj,
            device=self.device,
            edge_dropout_rate=self.edge_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
        )

    def forward(self):
        """A forward pass of the MixGCF"""
        user_gcn_emb, item_gcn_emb = self.gcn(
            self.user_embedding.weight,
            self.item_embedding.weight,
            edge_dropout=self.edge_dropout,
            mess_dropout=self.mess_dropout,
        )

        return user_gcn_emb, item_gcn_emb

    def pooling(self, embeddings):
        """All pooling functions including mean, sum and concate"""
        # [-1, n_hops, channel]
        if self.pool == "mean":
            return embeddings.mean(dim=1)
        elif self.pool == "sum":
            return embeddings.sum(dim=1)
        elif self.pool == "concat":
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def predict(self, users, items):
        """Model prediction"""
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)

        with torch.no_grad():

            user_gcn_emb, item_gcn_emb = self.gcn(
                self.user_embedding.weight,
                self.item_embedding.weight,
                edge_dropout=False,
                mess_dropout=False,
            )
            user_gcn_emb, item_gcn_emb = (
                self.pooling(user_gcn_emb),
                self.pooling(item_gcn_emb),
            )

            u_g_embeddings = user_gcn_emb[users_t]
            i_g_embeddings = item_gcn_emb[items_t]
            scores = torch.mul(u_g_embeddings, i_g_embeddings).sum(dim=1)

        return scores


class LightGCNEngine(ModelEngine):
    # A class includes train an epoch and train a batch of MixGCF

    def __init__(self, config):

        self.config = config
        self.norm_adj = config["model"]["norm_adj"]
        self.model = LightGCN(config["model"], self.norm_adj)
        self.decay = config["model"]["l2"]
        self.pool = config["model"]["pool"]
        self.n_negs = config["model"]["n_negs"]
        self.ns = config["model"]["ns"]
        self.K = config["model"]["K"]
        super(LightGCNEngine, self).__init__(config)
        self.model.to(self.device)

    def train_single_batch(self, batch_data):

        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        norm_adj = self.norm_adj

        batch_users, pos_items, neg_items = batch_data

        user_gcn_emb, item_gcn_emb = self.model.forward()

        if self.ns == "rns":  # n_negs = 1
            neg_gcn_embs = item_gcn_emb[neg_items[:, : self.K]]
        else:
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(
                    self.negative_sampling(
                        user_gcn_emb,
                        item_gcn_emb,
                        batch_users,
                        neg_items[:, k * self.n_negs : (k + 1) * self.n_negs],
                        pos_items,
                    )
                )
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        return self.create_bpr_loss(
            user_gcn_emb[batch_users], item_gcn_emb[pos_items], neg_gcn_embs
        )

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

    def negative_sampling(
        self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item
    ):
        """Negative sampling"""
        batch_size = user.shape[0]
        s_e, p_e = (
            user_gcn_emb[user],
            item_gcn_emb[pos_item],
        )  # [batch_size, n_hops+1, channel]
        if self.pool != "concat":
            s_e = self.model.pooling(s_e).unsqueeze(dim=1)

        """positive mixing"""
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(
            dim=-1
        )  # [batch_size, n_negs, n_hops+1]
        # indices = torch.max(scores, dim=1)[1].detach()
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute(
            [0, 2, 1, 3]
        )  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[
            [[i] for i in range(batch_size)], range(neg_items_emb_.shape[1]), indices, :
        ]

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        """Compute BPR loss"""
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = user_gcn_emb.shape[0]

        u_e = self.model.pooling(user_gcn_emb)
        pos_e = self.model.pooling(pos_gcn_embs)
        neg_e = self.model.pooling(
            neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])
        ).view(batch_size, self.K, -1)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(
            torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1
        )  # [batch_size, K]

        mf_loss = torch.mean(
            torch.log(
                1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)
            )
        )

        # cul regularizer
        regularize = (
            torch.norm(user_gcn_emb[:, 0, :]) ** 2
            + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
            + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2
        ) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss

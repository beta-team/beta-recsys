import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from beta_rec.models.torch_engine import ModelEngine


def randint_choice(high, size=None, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive)."""
    a = np.arange(high)
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = p.flatten()
        p[exclusion] = 0
    if p is not None:
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample


def l2_norm(user_emb1):
    norm = torch.norm(user_emb1, 2, 1)
    return (user_emb1.T / norm).T


class calc_ssl_loss(nn.Module):
    def __init__(self, ssl_reg, ssl_temp):
        super(calc_ssl_loss, self).__init__()
        self.ssl_reg = ssl_reg
        self.ssl_temp = ssl_temp

    def forward(
        self,
        ua_embeddings_sub1,
        ua_embeddings_sub2,
        ia_embeddings_sub1,
        ia_embeddings_sub2,
        users,
        pos_items,
    ):

        user_emb1 = ua_embeddings_sub1[users]
        user_emb2 = ua_embeddings_sub2[users]

        normalize_user_emb1 = l2_norm(user_emb1)
        normalize_user_emb2 = l2_norm(user_emb2)

        item_emb1 = ia_embeddings_sub1[pos_items]
        item_emb2 = ia_embeddings_sub2[pos_items]

        normalize_item_emb1 = l2_norm(item_emb1)
        normalize_item_emb2 = l2_norm(item_emb2)

        normalize_user_emb2_neg = normalize_user_emb2
        normalize_item_emb2_neg = normalize_item_emb2

        pos_score_user = torch.sum(
            torch.mul(normalize_user_emb1, normalize_user_emb2), dim=1
        )
        ttl_score_user = torch.matmul(normalize_user_emb1, normalize_user_emb2_neg.T)

        pos_score_item = torch.sum(
            torch.mul(normalize_item_emb1, normalize_item_emb2), dim=1
        )
        ttl_score_item = torch.matmul(normalize_item_emb1, normalize_item_emb2_neg.T)

        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.ssl_temp), dim=1)

        ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))
        ssl_loss_item = -torch.sum(torch.log(pos_score_item / ttl_score_item))
        ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)
        return ssl_loss


class calc_ssl_loss_v2(nn.Module):
    def __init__(self, ssl_reg, ssl_temp, ssl_mode):
        super(calc_ssl_loss_v2, self).__init__()

        self.ssl_reg = ssl_reg
        self.ssl_temp = ssl_temp
        self.ssl_mode = ssl_mode

    def forward(
        self,
        ua_embeddings_sub1,
        ua_embeddings_sub2,
        ia_embeddings_sub1,
        ia_embeddings_sub2,
        users,
        pos_items,
    ):

        if self.ssl_mode in ["user_side", "both_side"]:
            user_emb1 = ua_embeddings_sub1[users]
            user_emb2 = ua_embeddings_sub2[users]

            normalize_user_emb1 = l2_norm(user_emb1)
            normalize_user_emb2 = l2_norm(user_emb2)

            normalize_all_user_emb2 = l2_norm(ua_embeddings_sub2)
            pos_score_user = torch.sum(
                torch.mul(normalize_user_emb1, normalize_user_emb2), dim=1
            )
            ttl_score_user = torch.matmul(
                normalize_user_emb1, normalize_all_user_emb2.T
            )

            pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
            ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), dim=1)
            ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))

        if self.ssl_mode in ["item_side", "both_side"]:

            item_emb1 = ia_embeddings_sub1[pos_items]
            item_emb2 = ia_embeddings_sub2[pos_items]

            normalize_item_emb1 = l2_norm(item_emb1)
            normalize_item_emb2 = l2_norm(item_emb2)
            normalize_all_item_emb2 = l2_norm(ia_embeddings_sub2)

            pos_score_item = torch.sum(
                torch.mul(normalize_item_emb1, normalize_item_emb2), dim=1
            )
            ttl_score_item = torch.matmul(
                normalize_item_emb1, normalize_all_item_emb2.T
            )
            pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
            ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.ssl_temp), dim=1)
            ssl_loss_item = -torch.sum(torch.log(pos_score_item / ttl_score_item))

        if self.ssl_mode == "user_side":
            ssl_loss = self.ssl_reg * ssl_loss_user
        elif self.ssl_mode == "item_side":
            ssl_loss = self.ssl_reg * ssl_loss_item
        else:
            ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)

        return ssl_loss


class calc_ssl_loss_v3(nn.Module):
    def __init__(self, ssl_reg, ssl_temp):
        super(calc_ssl_loss_v3, self).__init__()
        self.ssl_reg = ssl_reg
        self.ssl_temp = ssl_temp

    def forward(
        self,
        ua_embeddings_sub1,
        ua_embeddings_sub2,
        ia_embeddings_sub1,
        ia_embeddings_sub2,
        users,
        pos_items,
    ):

        batch_users = torch.unique(users)

        user_emb1 = ua_embeddings_sub1[batch_users]
        user_emb2 = ua_embeddings_sub2[batch_users]
        batch_items, _ = torch.unique(pos_items)
        item_emb1 = ia_embeddings_sub1[batch_items]
        item_emb2 = ia_embeddings_sub2[batch_items]

        emb_merge1 = torch.cat([user_emb1, item_emb1], dim=0)
        emb_merge2 = torch.cat([user_emb2, item_emb2], dim=0)
        # cosine similarity
        normalize_emb_merge1 = l2_norm(emb_merge1)
        normalize_emb_merge2 = l2_norm(emb_merge2)
        pos_score = torch.sum(
            torch.mul(normalize_emb_merge1, normalize_emb_merge2), dim=1
        )
        ttl_score = torch.matmul(normalize_emb_merge1, normalize_emb_merge2.T)
        pos_score = torch.exp(pos_score / self.ssl_temp)
        ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
        ssl_loss = -torch.sum(torch.log(pos_score / ttl_score))
        ssl_loss = self.ssl_reg * ssl_loss
        return ssl_loss


class create_bpr_loss(nn.Module):
    def __init__(self, reg):
        super(create_bpr_loss, self).__init__()
        self.reg = reg

    def forward(
        self,
        ua_embeddings,
        ia_embeddings,
        users,
        pos_items,
        neg_items,
        user_embedding,
        item_embedding,
    ):

        batch_u_embeddings = ua_embeddings[users]
        batch_pos_i_embeddings = ia_embeddings[pos_items]
        batch_neg_i_embeddings = ia_embeddings[neg_items]

        batch_u_embeddings_pre = user_embedding[users]
        batch_pos_i_embeddings_pre = item_embedding[pos_items]
        batch_neg_i_embeddings_pre = item_embedding[neg_items]

        regularizer = (
            torch.sum(torch.pow(batch_u_embeddings_pre, 2)) / 2
            + torch.sum(torch.pow(batch_pos_i_embeddings_pre, 2)) / 2
            + torch.sum(torch.pow(batch_neg_i_embeddings_pre, 2)) / 2
        )
        emb_loss = self.reg * regularizer
        pos_scores = torch.sum(
            torch.mul(batch_u_embeddings, batch_pos_i_embeddings), dim=-1
        )
        neg_scores = torch.sum(
            torch.mul(batch_u_embeddings, batch_neg_i_embeddings), dim=-1
        )
        bpr_loss = torch.sum(-torch.log(torch.sigmoid(pos_scores - neg_scores)))

        return bpr_loss, emb_loss


class SGL(nn.Module):
    def __init__(self, config):
        self.config = config
        super(SGL, self).__init__()
        self.create_bpr_loss = create_bpr_loss(self.config["regs"])
        self.calc_ssl_loss_v3 = calc_ssl_loss_v3(
            self.config["ssl_reg"], self.config["ssl_temp"]
        )
        self.calc_ssl_loss_v2 = calc_ssl_loss_v2(
            self.config["ssl_reg"], self.config["ssl_temp"], self.config["ssl_mode"]
        )
        self.calc_ssl_loss = calc_ssl_loss(
            self.config["ssl_reg"], self.config["ssl_temp"]
        )
        self.ssl_mode = self.config["ssl_mode"]
        self.norm_adj = self.config["norm_adj"]

        self.n_users = self.config["n_users"]
        self.n_items = self.config["n_items"]
        self.n_layers = self.config["n_layers"]
        self.pretrain = self.config["pretrain"]

        self.ua_embeddings = None
        self.ia_embeddings = None
        self.embe_dim = self.config["emb_dim"]

        self.aug_type = self.config["aug_type"]

        self.user_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.n_users, self.embe_dim)),
            requires_grad=True,
        )
        self.item_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.n_items, self.embe_dim)),
            requires_grad=True,
        )
        self.adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col])
        return torch.sparse_coo_tensor(torch.tensor(indices), coo.data, coo.shape)

    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col])
        return indices, coo.data, coo.shape

    def forward(self, sub_mat, bat_users, bat_pos_items, bat_neg_items):

        for k in range(1, self.n_layers + 1):
            if self.aug_type in [0, 1]:
                sub_mat["sub_mat_1%d" % k] = torch.sparse_coo_tensor(
                    sub_mat["adj_indices_sub1"],
                    sub_mat["adj_values_sub1"],
                    sub_mat["adj_shape_sub1"],
                ).to(self.device)
                sub_mat["sub_mat_2%d" % k] = torch.sparse_coo_tensor(
                    sub_mat["adj_indices_sub2"],
                    sub_mat["adj_values_sub2"],
                    sub_mat["adj_shape_sub2"],
                ).to(self.device)
            else:
                sub_mat["sub_mat_1%d" % k] = torch.sparse_coo_tensor(
                    sub_mat["adj_indices_sub1%d" % k],
                    sub_mat["adj_values_sub1%d" % k],
                    sub_mat["adj_shape_sub1%d" % k],
                ).to(self.device)
                sub_mat["sub_mat_2%d" % k] = torch.sparse_coo_tensor(
                    sub_mat["adj_indices_sub2%d" % k],
                    sub_mat["adj_values_sub2%d" % k],
                    sub_mat["adj_shape_sub2%d" % k],
                ).to(self.device)

        ego_embeddings = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        ego_embeddings_sub1 = ego_embeddings
        ego_embeddings_sub2 = ego_embeddings
        all_embeddings = [ego_embeddings]
        all_embeddings_sub1 = [ego_embeddings_sub1]
        all_embeddings_sub2 = [ego_embeddings_sub2]

        for k in range(1, self.n_layers + 1):

            ego_embeddings = torch.matmul(self.adj_mat.to(self.device), ego_embeddings)
            all_embeddings += [ego_embeddings]

            ego_embeddings_sub1 = torch.matmul(
                sub_mat["sub_mat_1%d" % k], ego_embeddings_sub1
            )

            all_embeddings_sub1 += [ego_embeddings_sub1]
            ego_embeddings_sub2 = torch.matmul(
                sub_mat["sub_mat_2%d" % k], ego_embeddings_sub2
            )
            all_embeddings_sub2 += [ego_embeddings_sub2]

        all_embeddings = torch.stack(all_embeddings, 1)
        all_embeddings = torch.mean(all_embeddings, dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items], 0
        )
        all_embeddings_sub1 = torch.stack(all_embeddings_sub1, 1)
        all_embeddings_sub1 = torch.mean(all_embeddings_sub1, dim=1, keepdim=False)
        u_g_embeddings_sub1, i_g_embeddings_sub1 = torch.split(
            all_embeddings_sub1, [self.n_users, self.n_items], 0
        )

        all_embeddings_sub2 = torch.stack(all_embeddings_sub2, 1)
        all_embeddings_sub2 = torch.mean(all_embeddings_sub2, dim=1, keepdim=False)
        u_g_embeddings_sub2, i_g_embeddings_sub2 = torch.split(
            all_embeddings_sub2, [self.n_users, self.n_items], 0
        )

        (
            ua_embeddings,
            ia_embeddings,
            ua_embeddings_sub1,
            ia_embeddings_sub1,
            ua_embeddings_sub2,
            ia_embeddings_sub2,
        ) = (
            u_g_embeddings,
            i_g_embeddings,
            u_g_embeddings_sub1,
            i_g_embeddings_sub1,
            u_g_embeddings_sub2,
            i_g_embeddings_sub2,
        )

        self.ua_embeddings = ua_embeddings
        self.ia_embeddings = ia_embeddings

        if self.pretrain:
            ssl_loss = 0
        else:
            if self.ssl_mode in ["user_side", "item_side", "both_side"]:
                ssl_loss = self.calc_ssl_loss_v2(
                    ua_embeddings_sub1,
                    ua_embeddings_sub2,
                    ia_embeddings_sub1,
                    ia_embeddings_sub2,
                    bat_users,
                    bat_pos_items,
                )
            elif self.ssl_mode in ["merge"]:
                ssl_loss = self.calc_ssl_loss_v3(
                    ua_embeddings_sub1,
                    ua_embeddings_sub2,
                    ia_embeddings_sub1,
                    ia_embeddings_sub2,
                    bat_users,
                    bat_pos_items,
                )
            else:
                raise ValueError("Invalid ssl_mode!")
        sl_loss, emb_loss = self.create_bpr_loss(
            ua_embeddings,
            ia_embeddings,
            bat_users,
            bat_pos_items,
            bat_neg_items,
            self.user_embedding,
            self.item_embedding,
        )

        return sl_loss, emb_loss, ssl_loss

    def predict(self, users, items):

        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)

        with torch.no_grad():
            user_embed = self.ua_embeddings[users_t]
            items_embed = self.ia_embeddings[items_t]
            scores = torch.mul(user_embed, items_embed).sum(dim=1)
        return scores


class SGLEngine(ModelEngine):
    """SGLEngine Class."""

    # A class includes train an epoch and train a batch of SGL

    def __init__(self, config):
        """Initialize SGLEngine Class."""
        self.config = config
        self.norm_adj = config["model"]["norm_adj"]
        self.n_layers = self.config["model"]["n_layers"]
        self.aug_type = self.config["model"]["aug_type"]
        self.model = SGL(config["model"])
        super(SGLEngine, self).__init__(config)
        self.model.to(self.device)

    def train_single_batch(self, sub_mat, batch_data):
        """Train the model in a single batch.

        Args:
            batch_data (list): batch users, positive items and negative items.
        Return:
            loss (float): batch loss.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()

        batch_users, pos_items, neg_items = batch_data

        sl_loss, emb_loss, ssl_loss = self.model(
            sub_mat, batch_users, pos_items, neg_items
        )

        batch_loss = sl_loss + emb_loss + ssl_loss

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
        sub_mat = self.sub_mat_refresher(train_loader)
        for batch_data in train_loader:
            loss = self.train_single_batch(sub_mat, batch_data)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def create_sgl_mat(self, train_loader):
        """Create adjacent matirx from the user-item interaction matrix."""
        n_users = self.config["model"]["n_users"]
        n_items = self.config["model"]["n_items"]
        n_nodes = n_users + n_items
        is_subgraph = self.config["model"]["is_subgraph"]
        aug_type = self.config["model"]["aug_type"]
        ssl_ratio = self.config["model"]["ssl_ratio"]
        user_np = train_loader.dataset.user_tensor.cpu().numpy()
        item_np = train_loader.dataset.pos_item_tensor.cpu().numpy()
        if is_subgraph and aug_type in [0, 1, 2] and ssl_ratio > 0:
            # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk
            if aug_type == 0:
                drop_user_idx = self.randint_choice(
                    n_users, size=n_users * ssl_ratio, replace=False
                )
                drop_item_idx = self.randint_choice(
                    n_items, size=n_items * ssl_ratio, replace=False
                )
                indicator_user = np.ones(n_users, dtype=np.float32)
                indicator_item = np.ones(n_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.0
                indicator_item[drop_item_idx] = 0.0
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(user_np, dtype=np.float32), (user_np, item_np)),
                    shape=(n_users, n_items),
                )
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix(
                    (ratings_keep, (user_np_keep, item_np_keep + n_users)),
                    shape=(n_nodes, n_nodes),
                )
            if aug_type in [1, 2]:
                keep_idx = self.randint_choice(
                    len(user_np),
                    size=int(len(user_np) * (1 - ssl_ratio)),
                    replace=False,
                )
                user_keep_np = np.array(user_np)[keep_idx]
                item_keep_np = np.array(item_np)[keep_idx]
                ratings = np.ones_like(user_keep_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix(
                    (ratings, (user_keep_np, item_keep_np + n_users)),
                    shape=(n_nodes, n_nodes),
                )
        else:
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix(
                (ratings, (user_np, item_np + n_users)), shape=(n_nodes, n_nodes)
            )
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def randint_choice(self, high, size=None, replace=True, p=None, exclusion=None):
        """Return random integers from `0` (inclusive) to `high` (exclusive)."""
        a = np.arange(high)
        if exclusion is not None:
            if p is None:
                p = np.ones_like(a)
            else:
                p = np.array(p, copy=True)
            p = p.flatten()
            p[exclusion] = 0
        if p is not None:
            p = p / np.sum(p)
        sample = np.random.choice(a, size=size, replace=replace, p=p)

        return sample

    def sub_mat_refresher(self, train_loader):

        sub_mat = {}
        if self.aug_type in [0, 1]:
            (
                sub_mat["adj_indices_sub1"],
                sub_mat["adj_values_sub1"],
                sub_mat["adj_shape_sub1"],
            ) = self.convert_csr_to_sparse_tensor_inputs(
                self.create_sgl_mat(train_loader)
            )
            (
                sub_mat["adj_indices_sub2"],
                sub_mat["adj_values_sub2"],
                sub_mat["adj_shape_sub2"],
            ) = self.convert_csr_to_sparse_tensor_inputs(
                self.create_sgl_mat(train_loader)
            )
        else:
            for k in range(1, self.n_layers + 1):
                (
                    sub_mat["adj_indices_sub1%d" % k],
                    sub_mat["adj_values_sub1%d" % k],
                    sub_mat["adj_shape_sub1%d" % k],
                ) = self.convert_csr_to_sparse_tensor_inputs(
                    self.create_sgl_mat(train_loader)
                )
                (
                    sub_mat["adj_indices_sub2%d" % k],
                    sub_mat["adj_values_sub2%d" % k],
                    sub_mat["adj_shape_sub2%d" % k],
                ) = self.convert_csr_to_sparse_tensor_inputs(
                    self.create_sgl_mat(train_loader)
                )

        return sub_mat

    def convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col])
        return indices, coo.data, coo.shape

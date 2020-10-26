import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse

from beta_rec.models.torch_engine import ModelEngine


class NGCF(torch.nn.Module):
    """Model initialisation, embedding generation and prediction of NGCF."""

    def __init__(self, config, norm_adj):
        """Initialize NGCF Class."""
        super(NGCF, self).__init__()
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.layer_size = config["layer_size"]
        self.norm_adj = norm_adj
        self.n_layers = len(self.layer_size)
        self.dropout = nn.ModuleList()
        self.GC_weights = nn.ModuleList()
        self.Bi_weights = nn.ModuleList()
        self.dropout_list = list(config["mess_dropout"])
        self.layer_size = [self.emb_dim] + self.layer_size
        # Create GNN layers

        for i in range(self.n_layers):
            self.GC_weights.append(
                nn.Linear(self.layer_size[i], self.layer_size[i + 1])
            )
            self.Bi_weights.append(
                nn.Linear(self.layer_size[i], self.layer_size[i + 1])
            )
            self.dropout.append(nn.Dropout(self.dropout_list[i]))

        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        self.init_emb()

    def init_emb(self):
        """Initialize users and itmes' embeddings."""
        # Initialize users and items' embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, norm_adj):
        """Perform GNN function on users and item embeddings.

        Args:
            norm_adj (torch sparse tensor): the norm adjacent matrix of the user-item interaction matrix.
        Returns:
            u_g_embeddings (tensor): processed user embeddings.
            i_g_embeddings (tensor): processed item embeddings.
        """
        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_embedding.weight), dim=0
        )
        all_embeddings = [ego_embeddings]

        norm_adj = norm_adj.to(self.device)
        for i in range(self.n_layers):
            side_embeddings = sparse.mm(norm_adj, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_weights[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_weights[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )

        return u_g_embeddings, i_g_embeddings

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
            ua_embeddings, ia_embeddings = self.forward(self.norm_adj)
            u_g_embeddings = ua_embeddings[users_t]
            i_g_embeddings = ia_embeddings[items_t]
            scores = torch.mul(u_g_embeddings, i_g_embeddings).sum(dim=1)
        return scores


class NGCFEngine(ModelEngine):
    """NGCFEngine Class."""

    # A class includes train an epoch and train a batch of NGCF

    def __init__(self, config):
        """Initialize NGCFEngine Class."""
        self.config = config
        self.regs = config["model"]["regs"]  # reg is the regularisation
        self.decay = self.regs[0]
        self.batch_size = config["model"]["batch_size"]
        self.norm_adj = config["model"]["norm_adj"]
        self.model = NGCF(config["model"], self.norm_adj)
        super(NGCFEngine, self).__init__(config)
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
        norm_adj = self.norm_adj
        ua_embeddings, ia_embeddings = self.model.forward(norm_adj)

        batch_users, pos_items, neg_items = batch_data

        u_g_embeddings = ua_embeddings[batch_users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

        batch_loss.backward()
        self.optimizer.step()
        loss = batch_loss.item()
        return loss, batch_reg_loss

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
            loss, reg = self.train_single_batch(batch_data)
            total_loss += loss
            regularizer += reg
        print(f"[Training Epoch {epoch_id}], Loss {loss}, Regularizer {regularizer}")
        self.writer.add_scalar("model/loss", total_loss, epoch_id)
        self.writer.add_scalar("model/regularizer", regularizer, epoch_id)

    def bpr_loss(self, users, pos_items, neg_items):
        """Bayesian Personalised Ranking (BPR) pairwise loss function.

        Note that the sizes of pos_scores and neg_scores should be equal.

        Args:
            pos_scores (tensor): Tensor containing predictions for known positive items.
            neg_scores (tensor): Tensor containing predictions for sampled negative items.

        Returns:
            loss.
        """
        # Calculate BPR loss
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = (
            1.0 / 2 * (users ** 2).sum()
            + 1.0 / 2 * (pos_items ** 2).sum()
            + 1.0 / 2 * (neg_items ** 2).sum()
        )
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

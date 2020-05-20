import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
from beta_rec.models.torch_engine import Engine


class NGCF(torch.nn.Module):
    """Model initialisation, embedding generation and prediction of NGCF

    """

    def __init__(self, config):
        super(NGCF, self).__init__()
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.layer_size = config["layer_size"]
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
        # Initialise users and items' embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, norm_adj):
        """ Perform GNN function on users and item embeddings
        Args:
            norm_adj (torch sparse tensor): the norm adjacent matrix of the user-item interaction matrix
        Returns:
            u_g_embeddings (tensor): processed user embeddings
            i_g_embeddings (tensor): processed item embeddings
        """
        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_embedding.weight), dim=0
        )
        all_embeddings = [ego_embeddings]

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
        """ Model prediction: dot product of users and items embeddings
        Args:
            users (int):  user id
            items (int):  item id
        Return:
            scores (int): dot product
        """
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)

        with torch.no_grad():
            scores = torch.mul(
                self.user_embedding(users_t), self.item_embedding(items_t)
            ).sum(dim=1)
        return scores


class NGCFEngine(Engine):
    # A class includes train an epoch and train a batch of NGCF

    def __init__(self, config):
        self.config = config
        self.model = NGCF(config)
        self.regs = config["regs"]  # reg is the regularisation
        self.decay = self.regs[0]
        self.batch_size = config["batch_size"]
        self.norm_adj = config["norm_adj"]
        self.num_batch = config["num_batch"]

        super(NGCFEngine, self).__init__(config)

    def train_single_batch(self, batch_data):
        """
        Args:
            batch_data (list): batch users, positive items and negative items
        Return:
            loss (float): batch loss
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
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        """ Generate batch data for each batch
        Args:
            epoch_id (int):
            train_loader (function): user, pos_items and neg_items generator
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0.0

        n_batch = self.num_batch

        for idx in range(n_batch):
            users, pos_items, neg_items = train_loader.sample(self.batch_size)
            batch_data = (users, pos_items, neg_items)
            loss = self.train_single_batch(batch_data)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def bpr_loss(self, users, pos_items, neg_items):
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from beta_rec.models.torch_engine import ModelEngine


def truncated_normal_(tensor, mean=0, std=1):
    """Missing Doc."""
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class PairwiseGMF(nn.Module):
    """PairwiseGMF Class.

    Constructs the user/item memories and user/item external memory/outputs.

    Also add the embedding lookups.
    """

    def __init__(self, config):
        """Initialize PairwiseGMF Class."""
        super(PairwiseGMF, self).__init__()
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        # MemoryEmbed
        self.user_memory = nn.Embedding(self.n_users, self.emb_dim)
        truncated_normal_(self.user_memory.weight, std=0.01)
        self.user_memory.weight.requires_grad = True

        # ItemMemory
        self.item_memory = nn.Embedding(self.n_items, self.emb_dim)
        truncated_normal_(self.item_memory.weight, std=0.01)
        self.item_memory.weight.requires_grad = True

        self.v = nn.Linear(self.emb_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.v.weight)
        self.v.weight.requires_grad = True

    def forward(self, input_users, input_items, input_items_negative):
        """Train the model.

        Construct the model; main part of it goes here.
        """
        # [batch, embedding size]
        cur_user = self.user_memory(input_users)
        # Item memories a query
        cur_item = self.item_memory(input_items)
        cur_item_negative = self.item_memory(input_items_negative)

        pos_score = F.relu(self.v(cur_user * cur_item))
        neg_score = F.relu(self.v(cur_user * cur_item_negative))

        return pos_score, neg_score

    def predict(self):
        """Predict result with the model."""
        pass


class PairwiseGMFEngine(ModelEngine):
    """PairwiseGMFEngine Class."""

    def __init__(self, config):
        """Initialize PairwiseGMFEngine CLass."""
        self.config = config
        self.model = PairwiseGMF(config)
        self.regs = config["regs"]  # reg is the regularisation
        self.batch_size = config["batch_size"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        super(PairwiseGMFEngine, self).__init__(config)

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
        batch_users = torch.LongTensor(np.array(batch_users, dtype=np.int32)).to(
            self.device
        )
        pos_items = torch.LongTensor(np.array(pos_items, dtype=np.int32)).to(
            self.device
        )
        neg_items = torch.LongTensor(np.array(neg_items, dtype=np.int32)).to(
            self.device
        )

        pos_score, neg_score = self.model(batch_users, pos_items, neg_items)
        batch_loss = self.bpr_loss(pos_score, neg_score)

        for name, param in self.model.named_parameters():
            if name in ["v.weight"]:
                l2 = torch.sqrt(param.pow(2).sum())
                batch_loss += self.config["pretrain_l2_lambda"] * l2

        batch_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
        self.optimizer.step()
        loss = batch_loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        """Train the model in one epoch."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0.0

        progress = tqdm(
            enumerate(
                train_loader.cmn_train_loader(
                    self.batch_size, False, self.config["neg_count"]
                )
            ),
            dynamic_ncols=True,
            total=(train_loader.n_train * self.config["neg_count"]) // self.batch_size,
        )

        for k, batch in progress:
            users = batch[:, 0]
            pos_items = batch[:, 1]
            neg_items = batch[:, 2]
            batch_data = (users, pos_items, neg_items)
            loss = self.train_single_batch(batch_data)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def bpr_loss(self, pos_score, neg_score):
        """Bayesian Personalised Ranking (BPR) pairwise loss function.

        Note that the sizes of pos_scores and neg_scores should be equal.

        Args:
            pos_scores (tensor): Tensor containing predictions for known positive items.
            neg_scores (tensor): Tensor containing predictions for sampled negative items.

        Returns:
            loss.
        """
        difference = pos_score - neg_score
        eps = 1e-12
        loss = -1 * torch.log(torch.sigmoid(difference) + eps)
        return torch.mean(loss)

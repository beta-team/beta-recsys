import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..models.pairwise_gmf import truncated_normal_
from ..models.torch_engine import ModelEngine
from ..models.vlml import VariableLengthMemoryLayer


class CollaborativeMemoryNetwork(nn.Module):
    """CollaborativeMemoryNetwork Class."""

    def __init__(
        self, config, user_embeddings, item_embeddings, item_user_list, device
    ):
        """Initialize CollaborativeMemoryNetwork Class."""
        super(CollaborativeMemoryNetwork, self).__init__()

        self.config = config
        self.device = device
        self.emb_dim = config["emb_dim"]
        self.neighborhood = item_user_list
        self.max_neighbors = max([len(x) for x in item_user_list.values()])
        config["max_neighbors"] = self.max_neighbors
        # MemoryEmbed

        self.user_memory = nn.Embedding(
            user_embeddings.shape[0], user_embeddings.shape[1]
        )
        self.user_memory.weight = nn.Parameter(torch.from_numpy(user_embeddings))
        self.user_memory.weight.requires_grad = True

        # ItemMemory
        self.item_memory = nn.Embedding(
            item_embeddings.shape[0], item_embeddings.shape[1]
        )
        self.item_memory.weight = nn.Parameter(torch.from_numpy(item_embeddings))
        self.item_memory.weight.requires_grad = True

        # MemoryOutput
        self.user_output = nn.Embedding(
            user_embeddings.shape[0], user_embeddings.shape[1]
        )
        truncated_normal_(self.user_output.weight, std=0.01)
        self.user_output.weight.requires_grad = True

        self.mem_layer = VariableLengthMemoryLayer(
            2, self.config["emb_dim"], self.device
        )

        self.dense = nn.Linear(self.emb_dim * 2, self.emb_dim, bias=True)
        self.dense.weight.requires_grad = True
        self.dense.bias.requires_grad = True
        nn.init.kaiming_normal_(self.dense.weight)
        self.dense.bias.data.fill_(1.0)

        self.out = nn.Linear(self.emb_dim, 1, bias=False)
        self.out.weight.requires_grad = True
        nn.init.xavier_uniform_(self.out.weight)

    def output_module(self, input):
        """Missing Doc."""
        output = F.relu(self.dense(input))
        output = self.out(output)
        return output.squeeze()

    def forward(
        self,
        input_users,
        input_items,
        input_items_negative,
        input_neighborhoods,
        input_neighborhood_lengths,
        input_neighborhoods_negative,
        input_neighborhood_lengths_negative,
        evaluation=False,
    ):
        """Train the model."""
        # get embeddings from user memory
        cur_user = self.user_memory(input_users)
        # cur_user_output = self.user_output(input_users)

        # get embeddings from item memory
        cur_item = self.item_memory(input_items)

        # queries
        query = (cur_user, cur_item)

        # positive
        neighbor = self.mem_layer(
            query,
            self.user_memory(input_neighborhoods),
            self.user_output(input_neighborhoods),
            input_neighborhood_lengths,
            self.config["max_neighbors"],
        )[-1]["output"]

        score = self.output_module(torch.cat((cur_user * cur_item, neighbor), 1))

        if evaluation:
            return score

        cur_item_negative = self.item_memory(input_items_negative)
        neg_query = (cur_user, cur_item_negative)

        # negative
        neighbor_negative = self.mem_layer(
            neg_query,
            self.user_memory(input_neighborhoods_negative),
            self.user_output(input_neighborhoods_negative),
            input_neighborhood_lengths_negative,
            self.config["max_neighbors"],
        )[-1]["output"]

        negative_output = self.output_module(
            torch.cat((cur_user * cur_item_negative, neighbor_negative), 1)
        )

        return score, negative_output

    def predict(self, users, items):
        """Predict result with the model."""
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)
        user_embedding = self.user_memory(users_t)
        item_embedding = self.item_memory(items_t)

        with torch.no_grad():
            score = torch.mul(user_embedding, item_embedding).sum(dim=1)
        return score


class cmnEngine(ModelEngine):
    """CMN Engine."""

    def __init__(self, config, user_embeddings, item_embeddings, item_user_list):
        """Initialize CMN Engine."""
        self.config = config
        self.device = config["device_str"]
        self.model = CollaborativeMemoryNetwork(
            config, user_embeddings, item_embeddings, item_user_list, self.device
        )
        self.regs = config["regs"]  # reg is the regularisation
        self.batch_size = config["batch_size"]
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=config["lr"], momentum=self.config["momentum"]
        )
        super(cmnEngine, self).__init__(config)
        self.model.to(self.device)

    def train_single_batch(self, batch_data):
        """Train a single batch data.

        Train a single batch data.

        Args:
            batch_data (list): batch users, positive items and negative items.
        Return:
            loss (float): batch loss.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()

        (
            input_users,
            input_items,
            input_items_negative,
            input_neighborhoods,
            input_neighborhood_lengths,
            input_neighborhoods_negative,
            input_neighborhood_lengths_negative,
        ) = batch_data

        pos_score, neg_score = self.model(
            input_users,
            input_items,
            input_items_negative,
            input_neighborhoods,
            input_neighborhood_lengths,
            input_neighborhoods_negative,
            input_neighborhood_lengths_negative,
        )

        batch_loss = self.bpr_loss(pos_score, neg_score)

        for name, param in self.model.named_parameters():
            if name in [
                "mem_layer.hop_mapping.1.weight",
                "output_module.dense.weight",
                "output_module.out.weight",
            ]:
                l2 = torch.sqrt(param.pow(2).sum())
                batch_loss += self.config["training_l2_lambda"] * l2
        batch_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
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

        progress = tqdm(
            enumerate(
                train_loader.cmn_train_loader(
                    self.batch_size, True, self.config["neg_count"]
                )
            ),
            dynamic_ncols=True,
            total=(train_loader.n_train * self.config["neg_count"]) // self.batch_size,
        )
        for k, batch in progress:
            (
                ratings,
                pos_neighborhoods,
                pos_neighborhood_length,
                neg_neighborhoods,
                neg_neighborhood_length,
            ) = batch

            input_users = torch.LongTensor(np.array(ratings[:, 0], dtype=np.int32)).to(
                self.device
            )
            input_items = torch.LongTensor(np.array(ratings[:, 1], dtype=np.int32)).to(
                self.device
            )
            input_items_negative = torch.LongTensor(
                np.array(ratings[:, 2], dtype=np.int32)
            ).to(self.device)
            input_neighborhoods = torch.LongTensor(
                np.array(pos_neighborhoods, dtype=np.int32)
            ).to(self.device)
            input_neighborhood_lengths = torch.LongTensor(
                np.array(pos_neighborhood_length, dtype=np.int32)
            ).to(self.device)
            input_neighborhoods_negative = torch.LongTensor(
                np.array(neg_neighborhoods, dtype=np.int32)
            ).to(self.device)
            input_neighborhood_lengths_negative = torch.LongTensor(
                np.array(neg_neighborhood_length, dtype=np.int32)
            ).to(self.device)

            batch_data = (
                input_users,
                input_items,
                input_items_negative,
                input_neighborhoods,
                input_neighborhood_lengths,
                input_neighborhoods_negative,
                input_neighborhood_lengths_negative,
            )

            loss = self.train_single_batch(batch_data)
            total_loss += loss

        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def bpr_loss(self, pos_score, neg_score):
        """Calculate BPR loss."""
        difference = pos_score - neg_score
        eps = 1e-12
        loss = -1 * torch.log(torch.sigmoid(difference) + eps)

        return torch.mean(loss)

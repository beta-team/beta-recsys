import torch
import torch.nn as nn

from beta_rec.models.torch_engine import ModelEngine
from beta_rec.utils.common_util import timeit


class MLP(torch.nn.Module):
    """MLP Class."""

    def __init__(self, config):
        """Initialize MLP Class."""
        super(MLP, self).__init__()
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.n_layers = config["mlp_config"]["n_layers"]
        self.dropout = config["dropout"]
        self.latent_dim = self.emb_dim * (2 ** (self.n_layers)) // 2

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.init_weight()

        MLP_modules = []
        for i in range(self.n_layers):
            input_size = self.emb_dim * (2 ** (self.n_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*MLP_modules)
        self.affine_output = torch.nn.Linear(in_features=self.emb_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        """Train the model."""
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat(
            [user_embedding, item_embedding], dim=-1
        )  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def predict(self, user_indices, item_indices):
        """Predict result with the model."""
        user_indices = torch.LongTensor(user_indices).to(self.device)
        item_indices = torch.LongTensor(item_indices).to(self.device)
        with torch.no_grad():
            return self.forward(user_indices, item_indices)

    def init_weight(self):
        """Initialize weight in the model."""
        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_user.weight, std=0.01)


class MLPEngine(ModelEngine):
    """Engine for training & evaluating GMF model."""

    def __init__(self, config):
        """Initialize MLPEngine Class."""
        self.model = MLP(config["model"])
        self.loss = torch.nn.BCELoss()
        super(MLPEngine, self).__init__(config)
        self.model.to(self.device)

    def train_single_batch(self, users, items, ratings):
        """Train the model in a single batch.

        Args:
            batch_data (list): batch users, positive items and negative items.
        Return:
            loss (float): batch loss.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        users, items, ratings = (
            users.to(self.device),
            items.to(self.device),
            ratings.to(self.device),
        )
        self.optimizer.zero_grad()
        ratings_pred = self.model.forward(users, items)
        loss = self.loss(ratings_pred.view(-1), ratings)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    @timeit
    def train_an_epoch(self, train_loader, epoch_id):
        """Train the model in one epoch.

        Args:
            epoch_id (int): the number of epoch.
            train_loader (function): user, pos_items and neg_items generator.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, total_loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

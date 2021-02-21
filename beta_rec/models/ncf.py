import os

import torch
import torch.nn as nn

from beta_rec.models.gmf import GMF
from beta_rec.models.mlp import MLP
from beta_rec.models.torch_engine import ModelEngine
from beta_rec.utils.common_util import timeit


class NeuMF(torch.nn.Module):
    """NeuMF Class."""

    def __init__(self, config):
        """Initialize NeuMF Class."""
        super(NeuMF, self).__init__()
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.n_layers = config["mlp_config"]["n_layers"]
        self.dropout = config["dropout"]
        self.latent_dim_mlp = self.emb_dim * (2 ** (self.n_layers)) // 2
        self.latent_dim_gmf = self.emb_dim

        self.embedding_user_mlp = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim_mlp
        )
        self.embedding_item_mlp = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim_mlp
        )
        self.embedding_user_mf = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim_gmf
        )
        self.embedding_item_mf = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim_gmf
        )

        MLP_modules = []
        for i in range(self.n_layers):
            input_size = self.emb_dim * (2 ** (self.n_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*MLP_modules)
        self.affine_output = torch.nn.Linear(
            in_features=self.emb_dim * 2, out_features=1
        )
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        """Train the model."""
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat(
            [user_embedding_mlp, item_embedding_mlp], dim=-1
        )  # the concat latent vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def predict(self, user_indices, item_indices):
        """Predict the result with the model."""
        user_indices = torch.LongTensor(user_indices).to(self.device)
        item_indices = torch.LongTensor(item_indices).to(self.device)
        with torch.no_grad():
            return self.forward(user_indices, item_indices)

    def init_weight(self):
        """Initialize weight in the model."""
        pass


class NeuMFEngine(ModelEngine):
    """Engine for training & evaluating GMF model."""

    def __init__(self, config):
        """Initialize NeuMFEngine Class."""
        self.config = config
        self.model = NeuMF(config["model"])
        self.loss = torch.nn.BCELoss()
        super(NeuMFEngine, self).__init__(config)
        print(self.model)
        if self.config["model"]["model"] == "ncf_pre":
            self.load_pretrain_weights()
        else:
            self.init_weights()

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
        ratings_pred = self.model(users, items)
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
            # assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def init_weights(self):
        """Initialize weights in the model."""
        nn.init.normal_(self.model.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.model.embedding_item_mf.weight, std=0.01)
        nn.init.normal_(self.model.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.model.embedding_user_mlp.weight, std=0.01)
        for m1 in self.model.fc_layers:
            if isinstance(m1, nn.Linear):
                nn.init.xavier_uniform_(m1.weight)
        nn.init.kaiming_uniform_(
            self.model.affine_output.weight, a=1, nonlinearity="sigmoid"
        )

    def load_pretrain_weights(self):
        """Load weights from trained MLP model & GMF model."""
        # load GMF model
        gmf_model = GMF(self.config["model"])
        gmf_save_dir = os.path.join(
            self.config["system"]["model_save_dir"],
            self.config["model"]["gmf_config"]["save_name"],
        )
        self.resume_checkpoint(
            gmf_save_dir,
            gmf_model,
        )
        self.model.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.model.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

        # load MLP model
        mlp_model = MLP(self.config)
        mlp_save_dir = os.path.join(
            self.config["system"]["model_save_dir"],
            self.config["model"]["mlp_config"]["save_name"],
        )
        self.resume_checkpoint(
            mlp_save_dir,
            mlp_model,
        )
        self.model.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.model.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
        for (m1, m2) in zip(self.model.fc_layers, mlp_model.fc_layers):
            if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                m1.weight.data.copy_(m2.weight)
                m1.bias.data.copy_(m2.bias)

        self.model.affine_output.weight.data = 0.5 * torch.cat(
            [mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data],
            dim=-1,
        )
        self.model.affine_output.bias.data = 0.5 * (
            mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data
        )

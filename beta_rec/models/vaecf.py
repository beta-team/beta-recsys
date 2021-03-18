import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

from beta_rec.models.torch_engine import ModelEngine


class VAE(nn.Module):
    def __init__(self, z_dim, ae_structure, config):
        super(VAE, self).__init__()
        self.config = config
        act_fn = self.config["activation"]
        if act_fn == "sigmoid":
            self.act_fn = nn.Sigmoid()
        elif act_fn == "tanh":
            self.act_fn = nn.Tanh()
        elif act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "relu6":
            self.act_fn = nn.ReLU6()

        self.likelihood = self.config["likelihood"]
        self.EPS = 1e-10
        # Encoder
        self.encoder = nn.Sequential()
        for i in range(len(ae_structure) - 1):
            self.encoder.add_module(
                "fc{}".format(i), nn.Linear(ae_structure[i], ae_structure[i + 1])
            )
            self.encoder.add_module("act{}".format(i), self.act_fn)
        self.enc_mu = nn.Linear(ae_structure[-1], z_dim)  # mu
        self.enc_logvar = nn.Linear(ae_structure[-1], z_dim)  # logvar

        # Decoder
        ae_structure = [z_dim] + ae_structure[::-1]
        self.decoder = nn.Sequential()
        for i in range(len(ae_structure) - 1):
            self.decoder.add_module(
                "fc{}".format(i), nn.Linear(ae_structure[i], ae_structure[i + 1])
            )
            if i != len(ae_structure) - 2:
                self.decoder.add_module("act{}".format(i), self.act_fn)

    def encode(self, x):
        h = self.encoder(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def decode(self, z):
        h = self.decoder(z)
        if self.likelihood == "mult":
            return torch.softmax(h, dim=1)
        else:
            return torch.sigmoid(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, x_, mu, logvar, beta):
        # Likelihood
        ll_choices = {
            "mult": x * torch.log(x_ + self.EPS),
            "bern": x * torch.log(x_ + self.EPS)
            + (1 - x) * torch.log(1 - x_ + self.EPS),
            "gaus": -((x - x_) ** 2),
            "pois": x * torch.log(x_ + self.EPS) - x_,
        }

        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError("Supported likelihoods: {}".format(ll_choices.keys()))

        ll = torch.sum(ll, dim=1)

        # KL term
        std = torch.exp(0.5 * logvar)
        kld = -0.5 * (1 + 2.0 * torch.log(std) - mu.pow(2) - std.pow(2))
        kld = torch.sum(kld, dim=1)

        return torch.mean(beta * kld - ll)

    def predict(self, user_idx, item_idx):
        """Predcit result with the model.

        Args:
            users (int, or list of int):  user id(s).
            items (int, or list of int):  item id(s).
        Return:
            scores (int, or list of int): predicted scores of these user-item pairs.
        """
        x_u = csr_matrix(
            (np.ones(len(user_idx)), (user_idx, item_idx)),
            shape=(self.config["n_users"], self.config["n_items"]),
        )

        z_u, _ = self.encode(
            torch.tensor(x_u.A, dtype=torch.float32, device=self.device)
        )
        user_pred = self.decode(z_u).data.flatten()[item_idx]

        return user_pred


class VAECFEngine(ModelEngine):
    """VAECF engine"""

    def __init__(self, config):
        """Initialise VAECF engine class"""
        self.config = config
        self.model = VAE(
            z_dim=10,
            ae_structure=[config["model"]["n_items"]] + [20],
            config=config["model"],
        )
        self.batch_size = config["model"]["batch_size"]
        self.n_users = config["model"]["n_users"]
        self.n_items = config["model"]["n_items"]
        self.lr = config["model"]["lr"]
        self.beta = config["model"]["beta"]
        self.wd = config["model"]["weight_decay"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        super(VAECFEngine, self).__init__(config)
        self.model.to(self.device)

    def train_single_batch(self, batch_data):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        u_batch_, mu, logvar = self.model.forward(batch_data)
        batch_loss = self.model.loss(batch_data, u_batch_, mu, logvar, self.beta)
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss.item()

    def train_an_epoch(self, train_loader, epoch_id):
        self.model.train()
        total_loss = 0.0

        batch_size = self.batch_size
        user_indices, matrix = train_loader
        input_size = len(np.arange(len(user_indices)))
        batch_num = int(np.ceil(input_size / batch_size))
        indices = np.arange(len(user_indices))
        for b in range(batch_num):
            start_offset = batch_size * b
            end_offset = batch_size * b + batch_size
            end_offset = min(end_offset, len(indices))
            batch_ids = indices[start_offset:end_offset]
            uids = user_indices[batch_ids]
            u_batch = matrix[uids, :]
            u_batch.data = np.ones(len(u_batch.data))
            u_batch = u_batch.A
            u_batch = torch.tensor(u_batch, dtype=torch.float32, device=self.device)
            batch_loss = self.train_single_batch(u_batch)
            total_loss += batch_loss

        print(f"[Training Epoch {epoch_id}], Loss {total_loss}")
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

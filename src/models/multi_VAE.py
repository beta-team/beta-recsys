import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from models.torch_engine import Engine


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, config):
        super(MultiVAE, self).__init__()
        p_dims = config["p_dims"]
        q_dims = config["q_dims"] if "q_dims" in config else None
        dropout = config["dropout"] if "dropout" in config else 0.5
        self.p_dims = p_dims
        self.alpha = config["alpha"]
        self.rec_x = torch.tensor(np.zeros(config["n_users"],config["n_items"]))

        if q_dims:
            assert (
                q_dims[0] == p_dims[-1]
            ), "In and Out dimensions must equal to each other"
            assert (
                q_dims[-1] == p_dims[0]
            ), "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]
            print("q_dims:", q_dims)

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])
            ]
        )
        self.p_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])
            ]
        )

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        
        mu, logvar = self.encode(input[:,1:])
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        recon_x.clone()
        return recon_x, mu, logvar

    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, : self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1] :]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            
    def predict(self, users, items):
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            pre_vector,_,_ = self.forward(self, input):


def loss_function(recon_x, x, mu, logvar, alpha=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + alpha * KLD


class MultiVAEEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.config = config
        self.model = MultiVAE(config)
        super(MultiVAEEngine, self).__init__(config)

    def train_single_batch(self, batch_data, alpha, ratings=None):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        recon_x, mu, logvar = self.model.forward(batch_data)
        loss = loss_function(recon_x, batch_data, mu, logvar, alpha)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        #         with autograd.detect_anomaly():
        for batch_id, batch_data in enumerate(train_loader):
            loss = self.train_single_batch(batch_data, self.model.alpha)
            total_loss += loss
        total_loss = total_loss / self.config["batch_size"]
        print(
            "[Training Epoch {}], loss {} alpha: {} lr: {}".format(
                epoch_id, total_loss, self.model.alpha, self.config["lr"],
            )
        )
        self.writer.add_scalars(
            "model/loss", {"total_loss": total_loss,}, epoch_id,
        )
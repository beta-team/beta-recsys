import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from beta_rec.datasets.seq_data_utils import SeqDataset, collate_fn
from beta_rec.models.torch_engine import ModelEngine


class NARM(nn.Module):
    """Neural Attentive Session Based Recommendation Model Class.

    Args:
        n_items(int): the number of items.
        hidden_size(int): the hidden size of gru.
        embedding_dim(int): the dimension of item embedding.
        batch_size(int):
        n_layers(int): the number of gru layers.
    """

    def __init__(self, config):
        """Initialize NARM Class."""
        super(NARM, self).__init__()
        self.config = config
        self.n_items = config["dataset"]["n_items"]
        self.hidden_size = config["model"]["hidden_size"]
        self.batch_size = config["model"]["batch_size"]
        self.n_layers = config["model"]["n_layers"]
        self.dropout_input = config["model"]["dropout_input"]
        self.dropout_hidden = config["model"]["dropout_hidden"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_input)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_hidden)
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        # self.sf = nn.Softmax()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, seq, lengths):
        """Train the model."""
        hidden = self.init_hidden(seq.size(1))
        embs = self.emb_dropout(self.emb(seq))
        embs = pack_padded_sequence(embs, lengths)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)

        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(
            gru_out.size()
        )
        q2 = self.a_2(ht)

        mask = torch.where(
            seq.permute(1, 0) > 0,
            torch.tensor([1.0], device=self.device),
            torch.tensor([0.0], device=self.device),
        )
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(
            mask.size()
        )
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)

        item_embs = self.emb(torch.arange(self.n_items).to(self.device))
        scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0))
        # scores = self.sf(scores)

        return scores

    def init_hidden(self, batch_size):
        """Initialize hidden layers."""
        return torch.zeros(
            (self.n_layers, batch_size, self.hidden_size), requires_grad=True
        ).to(self.device)


class NARMEngine(ModelEngine):
    """Engine for training & evaluating NARM model."""

    def __init__(self, config):
        """Initialize NARMEngine Class."""
        self.config = config
        self.model = NARM(config)
        super(NARMEngine, self).__init__(config)
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.config["model"]["lr_dc_step"],
            gamma=self.config["model"]["lr_dc"],
        )
        self.loss_func = nn.CrossEntropyLoss()
        print(self.model)

    def train_an_epoch(self, train_loader, epoch):
        """Train the model in one epoch.

        Args:
            epoch_id (int): the number of epoch.
            train_loader (function): user, pos_items and neg_items generator.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"

        st = time.time()
        print("Start Epoch #", epoch)
        self.scheduler.step(epoch=epoch)

        self.model.train()
        losses = []

        for i, (seq, target, lens) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            seq = seq.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            logit = self.model(seq, lens)

            loss = self.loss_func(logit, target)
            losses.append(loss.item())
            loss.backward()

            self.optimizer.step()

        mean_loss = np.mean(losses)
        print(
            "Epoch: {}, train loss: {:.4f}, time: {}".format(
                epoch, mean_loss, time.time() - st
            )
        )
        self.writer.add_scalar("model/loss", mean_loss, epoch)

        return mean_loss

    def predict(self, user_profile, batch=1):
        """Predict the next item given user profile.

        Args:
            user_profile (List): Contains the item IDs of the events.
            batch (int): Prediction batch size.

        Returns:
            preds (List): Prediction scores for selected items for every event of the batch.
        """
        seq = [user_profile]
        labels = [[0]]  # fake label

        valid_data = (seq, labels)

        valid_data = SeqDataset(valid_data, print_info=False)

        valid_loader = DataLoader(
            valid_data, batch_size=batch, shuffle=False, collate_fn=collate_fn
        )

        self.model.eval()

        with torch.no_grad():
            for seq, target, lens in valid_loader:
                seq = seq.to(self.device)
                outputs = self.model(seq, lens)
                outputs = F.softmax(outputs, dim=1)
        preds = outputs.detach().cpu().numpy()[0]  # [1:]
        # print("preds:", preds)
        # print("lens:",len(preds))
        return preds

    def recommend(self, user_profile, user_id=None):
        """Make a recommendation.

        Args:
            user_profile (List): Contains the item IDs of the events.
            user_id (None): users' id for personalised recommenation.

        Returns:
            List: item and score pairs.
        """
        pred = self.predict(user_profile, batch=1)

        pred = pd.DataFrame(
            data=pred, index=np.arange(self.config["dataset"]["n_items"])
        )

        # sort items by predicted score
        pred.sort_values(0, ascending=False, inplace=True)

        # convert to the required output format
        return [([x.index], x._2) for x in pred.reset_index().itertuples()]

    @staticmethod
    def get_recommendation_list(recommendation):
        """Missing Doc."""
        return list(map(lambda x: x[0], recommendation))

    @staticmethod
    def get_recommendation_confidence_list(recommendation):
        """Missing Doc."""
        return list(map(lambda x: x[1], recommendation))

import numpy as np
import torch
import torch.nn as nn

from beta_rec.models.torch_engine import ModelEngine


class PointWiseFeedForward(torch.nn.Module):
    """PointWise forward Module."""

    def __init__(self, hidden_units, dropout_rate):
        """Class Initialization.

        Args:
            hidden_units ([int]): Embedding dimension.
            dropout_rate ([float]): dropout rate.
        """
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        """Forward functioin.

        Args:
            inputs ([type]): [description]

        Returns:
            [type]: [description]
        """
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(nn.Module):
    """SASRec Class."""

    def __init__(self, config):
        """Initialize SASRec Class."""
        super(SASRec, self).__init__()
        self.config = config
        self.user_num = config["n_users"]
        self.item_num = config["n_items"]
        self.hidden_units = config["emb_dim"]
        self.maxlen = config["maxlen"]
        self.num_blocks = config["num_blocks"]
        self.num_heads = config["num_heads"]
        self.dropout_rate = config["dropout_rate"]
        self.batch_size = config["batch_size"]
        self.l2_emb = config["l2_emb"]

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, self.hidden_units, padding_idx=0
        )
        self.pos_emb = torch.nn.Embedding(self.maxlen, self.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                self.hidden_units, self.num_heads, self.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        """Encode sequential items.

        Args:
            log_seqs ([type]): [description]

        Returns:
            [type]: [description]
        """
        seqs = self.item_emb(
            torch.as_tensor(log_seqs, dtype=torch.long).to(self.device)
        )
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(
            torch.as_tensor(positions, dtype=torch.long).to(self.device)
        )
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.device)
        )

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        """Forward functioin.

        Args:q
            user_ids ([type]): [description]
            log_seqs ([type]): [description]
            pos_seqs ([type]): [description]
            neg_seqs ([type]): [description]

        Returns:
            [type]: [pos_logits neg_logits]
        """
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(
            torch.as_tensor(pos_seqs, dtype=torch.long, device=self.device)
        )
        neg_embs = self.item_emb(
            torch.as_tensor(neg_seqs, dtype=torch.long, device=self.device)
        )

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        """Predict scores for input item sequential.

        Args:
            user_ids ([type]): [description]
            log_seqs ([type]): [description]
            item_indices ([type]): [description]

        Returns:
            [type]: [logits]
        """
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(
            torch.as_tensor(item_indices, dtype=torch.long, device=self.device)
        )  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)


class SASRecEngine(ModelEngine):
    """Engine for training Triple model."""

    def __init__(self, config):
        """Initialize Triple2vecEngine Class."""
        self.config = config
        print(config)
        self.model = SASRec(config["model"])
        self.num_batch = config["model"]["n_users"] // config["model"]["batch_size"]
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        super(SASRecEngine, self).__init__(config)

    def train_single_batch(self, batch_data, ratings=None):
        """Train the model in a single batch."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        u, seq, pos, neg = batch_data
        pos_logits, neg_logits = self.model(u, seq, pos, neg)
        pos_labels, neg_labels = (
            torch.ones(pos_logits.shape, device=self.device),
            torch.zeros(neg_logits.shape, device=self.device),
        )
        # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
        indices = np.where(pos != 0)
        loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += self.bce_criterion(neg_logits[indices], neg_labels[indices])
        for param in self.model.item_emb.parameters():
            loss += self.model.l2_emb * torch.norm(param)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, sampler, epoch_id):
        """Train the model in one epoch."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        for batch_id in range(self.num_batch):
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            batch_data = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            loss = self.train_single_batch(batch_data)
            # print(
            #     "loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())
            # )  # expected 0.4~0.6 after init few epochs
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, total_loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

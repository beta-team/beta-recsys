import numpy as np
import torch
import torch.nn as nn

from beta_rec.models.torch_engine import ModelEngine


class PointWiseFeedForward(torch.nn.Module):
    """PointWise forward Module.

    Args:
        torch ([type]): [description]
    """

    def __init__(self, hidden_units, dropout_rate):  # wried, why fusion X 2?
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
        """Forward function.

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


class TimeAwareMultiHeadAttention(torch.nn.Module):
    """TimeAwareMultiHeadAttention forward Module.

    Args:
        torch ([type]): [description]
    """

    def __init__(self, hidden_size, head_num, dropout_rate):
        """Class Initialization.

        Args:
            hidden_size ([type]): [description]
            head_num ([type]): [description]
            dropout_rate ([type]): [description]
        """
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate

    def forward(
        self,
        queries,
        keys,
        time_mask,
        attn_mask,
        time_matrix_K,
        time_matrix_V,
        abs_pos_K,
        abs_pos_V,
    ):
        """Forward function.

        Args:
            queries ([type]): [description]
            keys ([type]): [description]
            time_mask ([type]): [description]
            attn_mask ([type]): [description]
            time_matrix_K ([type]): [description]
            time_matrix_V ([type]): [description]
            abs_pos_K ([type]): [description]
            abs_pos_V ([type]): [description]

        Returns:
            [type]: [description]
        """
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(
            torch.split(time_matrix_K, self.head_size, dim=3), dim=0
        )
        time_matrix_V_ = torch.cat(
            torch.split(time_matrix_V, self.head_size, dim=3), dim=0
        )
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        # time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) * (
            -(2 ** 32) + 1
        )  # -1e23 # float('-inf')
        paddings = paddings.to("cuda")
        attn_weights = torch.where(
            time_mask, paddings, attn_weights
        )  # True:pick padding
        attn_weights = torch.where(
            attn_mask, paddings, attn_weights
        )  # enforcing causality

        attn_weights = self.softmax(
            attn_weights
        )  # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += (
            attn_weights.unsqueeze(2)
            .matmul(time_matrix_V_)
            .reshape(outputs.shape)
            .squeeze(2)
        )

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(
            torch.split(outputs, Q.shape[0], dim=0), dim=2
        )  # div batch_size

        return outputs


class TiSASRec(nn.Module):
    """Time Interval Aware Self-Attention for Sequential Recommendation class.

    Args:
        nn ([type]): [description]
    """

    def __init__(self, config):
        """Initialize TiSASRec Class.

        Args:
            config ([type]): [description]
        """
        super(TiSASRec, self).__init__()
        self.config = config
        self.user_num = config["n_users"]
        self.item_num = config["n_items"]
        self.hidden_units = config["emb_dim"]
        self.maxlen = config["maxlen"]
        self.time_span = config["time_span"]
        self.num_blocks = config["num_blocks"]
        self.num_heads = config["num_heads"]
        self.dropout_rate = config["dropout_rate"]
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.l2_emb = config["l2_emb"]
        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, self.hidden_units, padding_idx=0
        )
        self.item_emb_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.abs_pos_K_emb = torch.nn.Embedding(self.maxlen, self.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(self.maxlen, self.hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(
            self.time_span + 1, self.hidden_units
        )
        self.time_matrix_V_emb = torch.nn.Embedding(
            self.time_span + 1, self.hidden_units
        )

        self.item_emb_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(
                self.hidden_units, self.num_heads, self.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def seq2feats(self, user_ids, log_seqs, time_matrices):
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
        seqs = self.item_emb_dropout(seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.as_tensor(positions, dtype=torch.long).to(self.device)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrices = torch.as_tensor(time_matrices, dtype=torch.long).to(self.device)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.device)
        )

        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.attention_layernorms[i](
                seqs
            )  # PyTorch mha requires time first fmt
            mha_outputs = self.attention_layers[i](
                Q,
                seqs,
                timeline_mask,
                attention_mask,
                time_matrix_K,
                time_matrix_V,
                abs_pos_K,
                abs_pos_V,
            )
            seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(
        self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs
    ):  # for training
        """Forward function.

        Args:
            user_ids ([type]): [description]
            log_seqs ([type]): [description]
            pos_seqs ([type]): [description]
            neg_seqs ([type]): [description]

        Returns:
            [type]: [pos_logits neg_logits]
        """
        log_feats = self.seq2feats(
            user_ids, log_seqs, time_matrices
        )  # user_ids hasn't been used yet

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

    def predict(self, user_ids, log_seqs, time_matrices, item_indices):  # for inference
        """Predict scores for input item sequential.

        Args:
            user_ids ([type]): [description]
            log_seqs ([type]): [description]
            item_indices ([type]): [description]

        Returns:
            [type]: [logits]
        """
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(
            torch.as_tensor(item_indices, dtype=torch.long, device=self.device)
        )  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)


class TiSASRecEngine(ModelEngine):
    """Engine for training & evaluating TiSASRec model."""

    def __init__(self, config):
        """Initialize TiSASRecEngine Class."""
        self.config = config
        print(config)
        self.model = TiSASRec(config["model"])
        self.num_batch = config["model"]["n_users"] // config["model"]["batch_size"]
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        super(TiSASRecEngine, self).__init__(config)

    def train_single_batch(self, batch_data, ratings=None):
        """Train the model in a single batch."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        u, seq, time_seq, time_matrix, pos, neg = batch_data
        pos_logits, neg_logits = self.model(u, seq, time_matrix, pos, neg)
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
            (
                u,
                seq,
                time_seq,
                time_matrix,
                pos,
                neg,
            ) = sampler.next_batch()  # tuples to ndarray
            batch_data = (
                np.array(u),
                np.array(seq),
                np.array(time_seq),
                np.array(time_matrix),
                np.array(pos),
                np.array(neg),
            )
            loss = self.train_single_batch(batch_data)
            # print(
            #     "loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())
            # )  # expected 0.4~0.6 after init few epochs
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, total_loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

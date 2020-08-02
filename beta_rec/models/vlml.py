import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableLengthMemoryLayer(nn.Module):
    """VariableLengthMemoryLayer Class."""

    def __init__(self, hops, emb_dim, device):
        """Initialize VariableLenghtMemoryLayer Class."""
        super(VariableLengthMemoryLayer, self).__init__()

        self.hops = hops
        self.device = device
        self.emb_dim = emb_dim
        self.hop_mapping = {}
        for h in range(hops - 1):
            self.hop_mapping[str(h + 1)] = nn.Linear(
                self.emb_dim, self.emb_dim, bias=True
            )
            self.hop_mapping[str(h + 1)].weight.requires_grad = True
            self.hop_mapping[str(h + 1)].bias.requires_grad = True
            nn.init.kaiming_normal_(self.hop_mapping[str(h + 1)].weight)
            self.hop_mapping[str(h + 1)].bias.data.fill_(1.0)
        self.hop_mapping = nn.ModuleDict(self.hop_mapping)

    def mask_mod(self, inputs, mask_length, maxlen=None):
        """Use a memory mask.

        Apply a memory mask such that the values we mask result in being the
        minimum possible value we can represent with a float32.

        :param inputs: [batch size, length], dtype=tf.float32.
        :param memory_mask: [batch_size] shape Tensor of ints indicating the length of inputs.
        :param maxlen: Sets the maximum length of the sequence; if None, inferred from inputs.
        :returns: [batch size, length] dim Tensor with the mask applied.
        """
        # [batch_size, length] => Sequence Mask
        memory_mask = torch.arange(maxlen).to(self.device).expand(
            len(mask_length), maxlen
        ) < mask_length.unsqueeze(1)
        memory_mask = memory_mask.float()

        # num_remaining_memory_slots = torch.sum(memory_mask, 1)

        # Get the numerical limits of a float
        finfo = np.finfo(np.float32)
        kept_indices = memory_mask

        ignored_indices = memory_mask < 1
        ignored_indices = ignored_indices.float()
        lower_bound = finfo.max * kept_indices + finfo.min * ignored_indices
        slice_length = torch.max(mask_length)

        # Return the elementwise
        return torch.min(inputs[:, :slice_length], lower_bound[:, :slice_length])

    def apply_attention_memory(
        self, memory, output_memory, query, memory_mask=None, maxlen=None
    ):
        """Apply attention memory.

        Args:
            :param memory: [batch size, max length, embedding size], typically Matrix M.
            :param output_memory: [batch size, max length, embedding size], typically Matrix C.
            :param query: [batch size, embed size], typically u.
            :param memory_mask: [batch size] dim Tensor, the length of each sequence if variable length.
            :param maxlen: int/Tensor, the maximum sequence padding length; if None it infers based on the max of
                memory_mask.
            :returns: AttentionOutput
                 output: [batch size, embedding size].
                 weight: [batch size, max length], the attention weights applied to
                         the output representation.
        """
        query_expanded = query.unsqueeze(-1).transpose(2, 1)

        batched_dot_prod = query_expanded * memory
        scores = batched_dot_prod.sum(2)

        if memory_mask is not None:
            scores = self.mask_mod(scores, memory_mask, maxlen)

        attention = F.softmax(scores, dim=-1)
        probs_temp = attention.unsqueeze(1)
        c_temp = output_memory.transpose(2, 1)
        neighborhood = c_temp * probs_temp

        weighted_output = neighborhood.sum(2)

        return {"weight": attention, "output": weighted_output}

    def forward(self, query, memory, output_memory, seq_length, maxlen=32):
        """Train the model."""
        # find maximum length of sequences in this batch
        cur_max = torch.max(seq_length).item()
        # slice to max length
        memory = memory[:, :cur_max]
        output_memory = output_memory[:, :cur_max]

        user_query, item_query = query
        hop_outputs = []

        # hop 0
        # z = m_u + e_i
        z = user_query + item_query

        for hop_k in range(self.hops):
            # hop 1, ... , hop self.hops-1
            if hop_k == 0:
                memory_hop = self.apply_attention_memory(
                    memory, output_memory, z, seq_length, maxlen
                )
            else:
                z = F.relu(self.hop_mapping[str(hop_k)](z) + memory_hop["output"])

                # apply attention
                memory_hop = self.apply_attention_memory(
                    memory, output_memory, z, seq_length, maxlen
                )

            hop_outputs.append(memory_hop)

        return hop_outputs

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FFWEncoder(nn.Module):

    def __init__(self, layer_sizes, voc_size):
        super(FFWEncoder, self).__init__()

        emb_size = layer_sizes[0]
        self.embedding = nn.Embedding(voc_size, emb_size)

        seq = []
        prev_hidden_size = emb_size
        for i, size in enumerate(layer_sizes[1:]):
            seq.append(nn.Linear(prev_hidden_size, size))
            if i < len(layer_sizes) - 1:
                seq.append(nn.ReLU())
            prev_hidden_size = size
        self.net = nn.Sequential(*seq)

    def forward(self, input_ids, attention_mask, token_type_ids):
        h = self.embedding(input_ids)
        # not using torch.mean otherwise we would not exclude padding
        expanded_attention = attention_mask.unsqueeze(-1).repeat(1, 1, h.shape[-1])
        padded_h = h * expanded_attention
        result_pooling = torch.sum(padded_h, axis=1) / torch.sum(attention_mask)

        h_transformed = self.net(result_pooling)
        return h_transformed

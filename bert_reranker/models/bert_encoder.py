import torch.nn as nn
import torch.nn.functional as F
import torch


class BertEncoder(nn.Module):

    def __init__(self, bert, max_seq_len, emb_dim, pooling_type):
        super(BertEncoder, self).__init__()
        self.pooling_type = pooling_type

        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.bert = bert
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
        if self.pooling_type == 'cls':
            result_pooling = h[:, 0]
        elif self.pooling_type == 'avg':
            # not using torch.mean to exclude padding
            expanded_attention = attention_mask.unsqueeze(-1).repeat(1, 1, h.shape[-1])
            padded_h = h * expanded_attention
            result_pooling = torch.sum(padded_h, axis=1) / torch.sum(attention_mask)
        else:
            raise ValueError('pooling {} not supported.'.format(self.pooling_type))
        h_transformed = self.net(result_pooling)
        return F.normalize(h_transformed)
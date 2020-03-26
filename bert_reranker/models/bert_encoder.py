import torch.nn as nn
import torch.nn.functional as F

# TODO
default_bert_emb_dim_global = 768


class BertEncoder(nn.Module):

    def __init__(self, bert, max_seq_len, emb_dim = default_bert_emb_dim_global):
        super(BertEncoder, self).__init__()
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
        h_cls = h[:, 0]
        h_transformed = self.net(h_cls)
        return F.normalize(h_transformed)
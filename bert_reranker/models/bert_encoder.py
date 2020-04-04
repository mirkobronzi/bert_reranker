import logging

import torch.nn as nn
import torch.nn.functional as F
import torch

logger = logging.getLogger(__name__)


class BertEncoder(nn.Module):

    def __init__(self, bert, max_seq_len, freeze_bert, pooling_type, top_layer_sizes, dropout,
                 normalize_bert_encoder_result, bert_dropout=None):
        super(BertEncoder, self).__init__()

        self.pooling_type = pooling_type
        self.max_seq_len = max_seq_len
        self.bert = bert
        if bert_dropout is not None:
            logger.info('setting bert dropout to {}'.format(bert_dropout))
            bert.config.attention_probs_dropout_prob = bert_dropout
            bert.config.hidden_dropout_prob = bert_dropout
        else:
            logger.info('using the original bert model dropout')

        self.freeze_bert = freeze_bert
        self.normalize_bert_encoder_result = normalize_bert_encoder_result
        seq = []
        prev_hidden_size = bert.config.hidden_size
        for i, size in enumerate(top_layer_sizes):
            seq.append(nn.Linear(prev_hidden_size, size))
            if i < len(top_layer_sizes) - 1:
                seq.append(nn.ReLU())
                seq.append(nn.Dropout(p=dropout, inplace=False))
            prev_hidden_size = size
        self.net = nn.Sequential(*seq)

    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.freeze_bert:
            with torch.no_grad():
                h, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        else:
            h, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        if self.pooling_type == 'cls':
            result_pooling = h[:, 0, :]
        elif self.pooling_type == 'avg':
            # not using torch.mean otherwise we would not exclude padding
            expanded_attention = attention_mask.unsqueeze(-1).repeat(1, 1, h.shape[-1])
            padded_h = h * expanded_attention
            result_pooling = torch.sum(padded_h, axis=1) / torch.sum(attention_mask)
        else:
            raise ValueError('pooling {} not supported.'.format(self.pooling_type))
        h_transformed = self.net(result_pooling)
        if self.normalize_bert_encoder_result:
            return F.normalize(h_transformed)
        else:
            return h_transformed


import logging

import torch.nn as nn
import torch.nn.functional as F
import torch

logger = logging.getLogger(__name__)


def compute_average_with_padding(tensor, padding):
    batch_size, seq_length, emb_size = tensor.shape
    expanded_padding = padding.unsqueeze(-1).repeat(1, 1, emb_size)
    padded_tensor = tensor * expanded_padding
    return torch.sum(padded_tensor, axis=1) / torch.sum(padding, axis=1).unsqueeze(1).repeat(1, emb_size)


def _get_layers(prev_hidden_size, dropout, layer_sizes, append_relu_and_dropout_after_last_layer):
    result = []
    for i, size in enumerate(layer_sizes):
        result.append(nn.Linear(prev_hidden_size, size))
        if i < len(layer_sizes) - 1 or append_relu_and_dropout_after_last_layer:
            result.append(nn.ReLU())
            result.append(nn.Dropout(p=dropout, inplace=False))
        prev_hidden_size = size
    return result


class BertEncoder(nn.Module):

    def __init__(self, bert, max_seq_len, freeze_bert, pooling_type, layers_pre_pooling,
                 layers_post_pooling, dropout, normalize_bert_encoder_result, bert_dropout=None):
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

        best_hidden_size = bert.config.hidden_size
        pre_pooling_seq = _get_layers(best_hidden_size, dropout, layers_pre_pooling, True)
        self.pre_pooling_net = nn.Sequential(*pre_pooling_seq)

        last_hidden_size = layers_pre_pooling[-1] if layers_pre_pooling else best_hidden_size
        post_pooling_seq = _get_layers(last_hidden_size, dropout, layers_post_pooling, False)
        self.post_pooling_net = nn.Sequential(*post_pooling_seq)

    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.freeze_bert:
            with torch.no_grad():
                bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        else:
            bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)

        pre_pooling_hs = self.pre_pooling_net(bert_hs)
        if self.pooling_type == 'cls':
            result_pooling = pre_pooling_hs[:, 0, :]
        elif self.pooling_type == 'avg':
            result_pooling = compute_average_with_padding(pre_pooling_hs, attention_mask)
        else:
            raise ValueError('pooling {} not supported.'.format(self.pooling_type))
        post_pooling_hs = self.post_pooling_net(result_pooling)

        if self.normalize_bert_encoder_result:
            return F.normalize(post_pooling_hs)
        else:
            return post_pooling_hs


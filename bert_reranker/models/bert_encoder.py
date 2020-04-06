import logging

import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import DistilBertModel, T5Model
from transformers import AutoModel

from bert_reranker.utils.hp_utils import check_and_log_hp

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

    def __init__(self, hyper_params, type):
        super(BertEncoder, self).__init__()

        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['bert_base', 'layers_pre_pooling', 'layers_post_pooling', 'dropout',
             'normalize_bert_encoder_result', 'dropout_bert', 'freeze_bert', 'pooling_type'],
            model_hparams)

        if type == 'question':
            self.max_seq_len = hyper_params['max_question_len']
        elif type == 'paragraph':
            self.max_seq_len = hyper_params['max_paragraph_len']
        else:
            raise ValueError('type {} not supported'.format(type))

        self.bert = AutoModel.from_pretrained(model_hparams['bert_base'])
        self.pooling_type = model_hparams['pooling_type']
        bert_dropout = model_hparams['dropout_bert']
        if bert_dropout is not None:
            logger.info('setting bert dropout to {}'.format(bert_dropout))
            self.bert.config.attention_probs_dropout_prob = bert_dropout
            self.bert.config.hidden_dropout_prob = bert_dropout
        else:
            logger.info('using the original bert model dropout')

        self.freeze_bert = model_hparams['freeze_bert']
        self.normalize_bert_encoder_result = model_hparams['normalize_bert_encoder_result']

        best_hidden_size = self.bert.config.hidden_size
        pre_pooling_seq = _get_layers(best_hidden_size, model_hparams['dropout'],
                                      model_hparams['layers_pre_pooling'],
                                      True)
        self.pre_pooling_net = nn.Sequential(*pre_pooling_seq)

        last_hidden_size = model_hparams['layers_pre_pooling'][-1] if \
            model_hparams['layers_pre_pooling'] else best_hidden_size
        post_pooling_seq = _get_layers(last_hidden_size, model_hparams['dropout'],
                                       model_hparams['layers_post_pooling'],
                                       False)
        self.post_pooling_net = nn.Sequential(*post_pooling_seq)

    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.freeze_bert:
            with torch.no_grad():
# <<<<<<< HEAD
                bert_hs = self.run_bert(attention_mask, input_ids, token_type_ids)
        else:
            bert_hs = self.run_bert(attention_mask, input_ids, token_type_ids)

# =======
#                 bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
#                                        token_type_ids=token_type_ids)
#         else:
#             bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
#                                    token_type_ids=token_type_ids)
#
#         pre_pooling_hs = self.pre_pooling_net(bert_hs)
# >>>>>>> different_lr_for_bert
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

    def run_bert(self, attention_mask, input_ids, token_type_ids):
        if type(self.bert) in {DistilBertModel, T5Model}:
            h = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            h = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
        return h[0]

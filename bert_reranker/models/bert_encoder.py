import logging

import torch
import torch.nn as nn
from transformers import AutoModel

from bert_reranker.models.general_encoder import GeneralEncoder
from bert_reranker.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


def get_ffw_layers(
        prev_hidden_size, dropout, layer_sizes, append_relu_and_dropout_after_last_layer):
    result = []
    for i, size in enumerate(layer_sizes):
        result.append(nn.Linear(prev_hidden_size, size))
        if i < len(layer_sizes) - 1 or append_relu_and_dropout_after_last_layer:
            result.append(nn.ReLU())
            result.append(nn.Dropout(p=dropout, inplace=False))
        prev_hidden_size = size
    return result


class BertEncoder(GeneralEncoder):

    def __init__(self, hyper_params):
        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['bert_base', 'dropout_bert', 'freeze_bert'],
            model_hparams)
        bert = AutoModel.from_pretrained(model_hparams['bert_base'])
        super(BertEncoder, self).__init__(hyper_params, bert.config.hidden_size)
        self.bert = bert
        bert_dropout = model_hparams['dropout_bert']
        if bert_dropout is not None:
            logger.info('setting bert dropout to {}'.format(bert_dropout))
            self.bert.config.attention_probs_dropout_prob = bert_dropout
            self.bert.config.hidden_dropout_prob = bert_dropout
        else:
            logger.info('using the original bert model dropout')

        self.freeze_bert = model_hparams['freeze_bert']

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids):
        if self.freeze_bert:
            with torch.no_grad():
                bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        else:
            bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        return bert_hs

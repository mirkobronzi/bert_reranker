import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_reranker.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


def compute_average_with_padding(tensor, padding):
    """

    :param tensor: dimension batch_size, seq_length, hidden_size
    :param padding: dimension batch_size, seq_length
    :return:
    """
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


class GeneralEncoder(nn.Module):

    def __init__(self, hyper_params, encoder_hidden_size):
        super(GeneralEncoder, self).__init__()

        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['layers_pre_pooling', 'layers_post_pooling', 'dropout',
             'normalize_model_result', 'pooling_type'],
            model_hparams)

        self.pooling_type = model_hparams['pooling_type']
        self.normalize_model_result = model_hparams['normalize_model_result']

        pre_pooling_seq = _get_layers(encoder_hidden_size, model_hparams['dropout'],
                                      model_hparams['layers_pre_pooling'],
                                      True)
        self.pre_pooling_net = nn.Sequential(*pre_pooling_seq)

        last_hidden_size = model_hparams['layers_pre_pooling'][-1] if \
            model_hparams['layers_pre_pooling'] else encoder_hidden_size
        post_pooling_seq = _get_layers(last_hidden_size, model_hparams['dropout'],
                                       model_hparams['layers_post_pooling'],
                                       False)
        self.post_pooling_net = nn.Sequential(*post_pooling_seq)

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids):
        raise ValueError('not implemented - use a subclass')

    def forward(self, input_ids, attention_mask, token_type_ids):
        hs = self.get_encoder_hidden_states(input_ids=input_ids, attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)

        pre_pooling_hs = self.pre_pooling_net(hs)

        if self.pooling_type == 'cls':
            result_pooling = pre_pooling_hs[:, 0, :]
        elif self.pooling_type == 'avg':
            result_pooling = compute_average_with_padding(pre_pooling_hs, attention_mask)
        else:
            raise ValueError('pooling {} not supported.'.format(self.pooling_type))

        post_pooling_hs = self.post_pooling_net(result_pooling)

        if self.normalize_model_result:
            return F.normalize(post_pooling_hs)
        else:
            return post_pooling_hs


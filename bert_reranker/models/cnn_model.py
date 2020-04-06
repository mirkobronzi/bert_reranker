import logging

import torch
import torch.nn as nn

from bert_reranker.models.general_encoder import GeneralEncoder
from bert_reranker.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class CNNEncoder(GeneralEncoder):

    def __init__(self, hyper_params, voc_size):
        raise ValueError('need to understand how to handle padding in the cnn layers')
        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['cnn_layer_sizes', 'emb_size'],
            model_hparams)
        super(CNNEncoder, self).__init__(hyper_params, hyper_params['model']['cnn_layer_sizes'][-1])

        emb_size = hyper_params['model']['emb_size']
        self.embedding = nn.Embedding(voc_size, emb_size)
        self.cnn = nn.Conv1d(emb_size, hyper_params['model']['cnn_layer_sizes'][-1], 10, stride=2)

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids):
        embs = self.embedding(input_ids)
        # cnn method wants batch_dim, channel_dim, seq_dim
        t_embs = torch.transpose(embs, 1, 2)
        result = self.cnn(t_embs)
        return torch.transpose(result, 1, 2)

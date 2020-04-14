import logging

import torch
import torch.nn as nn

from bert_reranker.models.general_encoder import GeneralEncoder
from bert_reranker.utils.hp_utils import check_and_log_hp

import pdb

logger = logging.getLogger(__name__)


class RNNEncoder(GeneralEncoder):

    def __init__(self, hyper_params, voc_size):
        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['num_layers', 'emb_size', 'rnn_latentsize'],
            model_hparams)

        super(RNNEncoder, self).__init__(hyper_params, 2*model_hparams['rnn_latentsize'])

        emb_size = hyper_params['model']['emb_size']
        self.embedding = nn.Embedding(voc_size, emb_size)
        self.rnn = nn.LSTM(emb_size, 
                           model_hparams['rnn_latentsize'], 
                           model_hparams['num_layers'], 
                           batch_first=True, 
                           dropout=model_hparams['dropout'], 
                           bidirectional=True)
        self.rnn.flatten_parameters()

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids):
        embs = self.embedding(input_ids)

        self.rnn.flatten_parameters()

        result = self.rnn(embs)[0]
        
        return result   # torch.transpose(result, 1, 2)

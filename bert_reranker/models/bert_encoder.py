import logging
import pickle

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


def hashable(input_id):
    return tuple(input_id.cpu().numpy())


class BertEncoder(GeneralEncoder):

    def __init__(self, hyper_params, name=''):
        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['bert_base', 'dropout_bert', 'freeze_bert'],
            model_hparams)
        bert = AutoModel.from_pretrained(model_hparams['bert_base'])
        super(BertEncoder, self).__init__(hyper_params, bert.config.hidden_size)
        self.bert = bert
        self.name = name

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


class CachedBertEncoder(BertEncoder):

    def __init__(self, hyper_params, name=''):
        model_hparams = hyper_params['model']
        check_and_log_hp(
            ['bert_base', 'dropout_bert', 'freeze_bert', 'cache_size'],
            model_hparams)
        super(CachedBertEncoder, self).__init__(hyper_params, name=name)

        if not model_hparams['freeze_bert'] or not model_hparams['dropout_bert'] == 0.0:
            raise ValueError('to cache results, set freeze_bert=True and dropout_bert=0.0')
        self.cache = {}
        self.cache_hit = 0
        self.cache_miss = 0
        self.max_cache_size = model_hparams['cache_size']

    def _search_in_cache(self, input_ids, attention_mask, token_type_ids):
        results = []
        still_to_compute_iids = []
        still_to_compute_am = []
        still_to_compute_tti = []
        for i in range(input_ids.shape[0]):
            ids_hash = hashable(input_ids[i])
            if ids_hash in self.cache:
                results.append(self.cache[ids_hash].to(input_ids.device))
            else:
                results.append(None)
                still_to_compute_iids.append(input_ids[i])
                still_to_compute_am.append(attention_mask[i])
                still_to_compute_tti.append(token_type_ids[i])
        return results, still_to_compute_iids, still_to_compute_am, still_to_compute_tti

    def _store_in_cache_and_get_results(self, cache_results, bert_hs, still_to_compute_iids):
        final_results = []
        non_cached_result_index = 0
        for cache_result in cache_results:
            if cache_result is None:
                non_cached_result = bert_hs[non_cached_result_index]
                final_results.append(non_cached_result)
                if len(self.cache) < self.max_cache_size:
                    self.cache[hashable(still_to_compute_iids[non_cached_result_index])] = \
                        non_cached_result.cpu()
                non_cached_result_index += 1
            else:
                final_results.append(cache_result)
        assert non_cached_result_index == bert_hs.shape[0]
        return torch.stack(final_results, dim=0)

    def get_encoder_hidden_states(self, input_ids, attention_mask, token_type_ids):

        cache_results, still_to_compute_iids, still_to_compute_am, still_to_compute_tti = \
            self._search_in_cache(input_ids, attention_mask, token_type_ids)
        self.cache_hit += input_ids.shape[0] - len(still_to_compute_iids)
        self.cache_miss += len(still_to_compute_iids)
        if len(still_to_compute_iids) == 0:
            return torch.stack(cache_results, dim=0)

        input_ids = torch.stack(still_to_compute_iids, dim=0)
        attention_mask = torch.stack(still_to_compute_am, dim=0)
        token_type_ids = torch.stack(still_to_compute_tti, dim=0)

        if self.freeze_bert:
            with torch.no_grad():
                bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        else:
            bert_hs, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)

        if self.cache is not None:
            bert_hs = self._store_in_cache_and_get_results(
                cache_results, bert_hs, still_to_compute_iids)

        return bert_hs

    def save_cache(self, save_to):
        with open(save_to, "wb") as out_stream:
            pickle.dump(self.cache, out_stream)

    def load_cache(self, load_from):
        with open(load_from, "rb") as in_stream:
            self.cache = pickle.load(in_stream)
        return len(self.cache)

    def print_stats_to(self, print_function):
        print_function('{}: cache size {} / cache hits {} / cache misses {}'.format(
            self.name, len(self.cache), self.cache_hit, self.cache_miss))

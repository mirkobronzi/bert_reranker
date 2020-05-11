import logging

from bert_reranker.models.bert_encoder import BertEncoder, CachedBertEncoder
from bert_reranker.models.cnn_model import CNNEncoder
from bert_reranker.models.retriever import EmbeddingRetriever, FeedForwardRetriever
from bert_reranker.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


def load_model(hyper_params, tokenizer, debug):
    check_and_log_hp(['name', 'single_encoder'], hyper_params['model'])
    if hyper_params['model']['name'] == 'bert_encoder':
        if hyper_params['model'].get('cache_size', 0) > 0:
            encoder_class = CachedBertEncoder
        else:
            encoder_class = BertEncoder

        bert_paragraph_encoder, bert_question_encoder = _create_encoders(encoder_class,
                                                                         hyper_params)

        model = EmbeddingRetriever(
            bert_question_encoder, bert_paragraph_encoder, tokenizer,
            hyper_params['max_question_len'], hyper_params['max_paragraph_len'], debug)
    elif hyper_params['model']['name'] == 'bert_ffw':

        if hyper_params['model'].get('cache_size', 0) > 0:
            encoder_class = CachedBertEncoder
        else:
            encoder_class = BertEncoder

        bert_paragraph_encoder, bert_question_encoder = _create_encoders(encoder_class,
                                                                         hyper_params)

        if bert_question_encoder.post_pooling_last_hidden_size != \
                bert_paragraph_encoder.post_pooling_last_hidden_size:
            raise ValueError("question/paragraph encoder should have the same output hidden size")
        previous_hidden_size = bert_question_encoder.post_pooling_last_hidden_size
        model = FeedForwardRetriever(
            bert_question_encoder, bert_paragraph_encoder, tokenizer,
            hyper_params['max_question_len'], hyper_params['max_paragraph_len'], debug,
            hyper_params['model'], previous_hidden_size=previous_hidden_size)
    elif hyper_params['model']['name'] == 'cnn':
        cnn_question_encoder = CNNEncoder(hyper_params, tokenizer.vocab_size)
        cnn_paragraph_encoder = CNNEncoder(hyper_params, tokenizer.vocab_size)
        model = EmbeddingRetriever(
            cnn_question_encoder, cnn_paragraph_encoder, tokenizer,
            hyper_params['max_question_len'], hyper_params['max_paragraph_len'], debug)
    else:
        raise ValueError('model name {} not supported'.format(hyper_params['model']['name']))
    return model


def _create_encoders(encoder, hyper_params):
    if hyper_params['model']['single_encoder']:
        logger.info('using a single BERT for both questions and answers')
        bert_question_encoder = encoder(
            hyper_params, bert_model=None, name='question')
        bert_paragraph_encoder = encoder(
            hyper_params, bert_model=bert_question_encoder.bert, name='paragraph')
    else:
        logger.info('using 2 BERT models: one for questions and one for answers')
        bert_question_encoder = encoder(hyper_params, bert_model=None, name='question')
        bert_paragraph_encoder = encoder(hyper_params, bert_model=None, name='paragraph')
    return bert_paragraph_encoder, bert_question_encoder

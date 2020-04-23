from bert_reranker.models.bert_encoder import BertEncoder, CachedBertEncoder
from bert_reranker.models.cnn_model import CNNEncoder
from bert_reranker.models.retriever import EmbeddingRetriever, FeedForwardRetriever
from bert_reranker.utils.hp_utils import check_and_log_hp


def load_model(hyper_params, tokenizer, debug):
    check_and_log_hp(['name'], hyper_params['model'])
    if hyper_params['model']['name'] == 'bert_encoder':
        if hyper_params['model'].get('cache_size', 0) > 0:
            encoder = CachedBertEncoder
        else:
            encoder = BertEncoder

        bert_question_encoder = encoder(hyper_params, name='question')
        bert_paragraph_encoder = encoder(hyper_params, name='paragraph')
        model = EmbeddingRetriever(
            bert_question_encoder, bert_paragraph_encoder, tokenizer,
            hyper_params['max_question_len'], hyper_params['max_paragraph_len'], debug)
    elif hyper_params['model']['name'] == 'bert_ffw':
        if hyper_params['model'].get('cache_size', 0) > 0:
            encoder = CachedBertEncoder
        else:
            encoder = BertEncoder

        bert_question_encoder = encoder(hyper_params, name='question')
        bert_paragraph_encoder = encoder(hyper_params, name='paragraph')
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

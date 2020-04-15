from bert_reranker.models.bert_encoder import BertEncoder
from bert_reranker.models.cnn_model import CNNEncoder
from bert_reranker.models.rnn_model import RNNEncoder

from bert_reranker.models.retriever import Retriever, EmbeddingRetriever, FeedForwardRetriever
from bert_reranker.utils.hp_utils import check_and_log_hp


def load_model(hyper_params, tokenizer, debug):
    check_and_log_hp(['name'], hyper_params['model'])
    if hyper_params['model']['name'] == 'bert_encoder':
        bert_question_encoder = BertEncoder(hyper_params)
        bert_paragraph_encoder = BertEncoder(hyper_params)
        model = EmbeddingRetriever(
            bert_question_encoder, bert_paragraph_encoder, tokenizer,
            hyper_params['max_question_len'], hyper_params['max_paragraph_len'], debug)
    elif hyper_params['model']['name'] == 'bert_ffw':
        bert_question_encoder = BertEncoder(hyper_params)
        bert_paragraph_encoder = BertEncoder(hyper_params)
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
        model = Retriever(cnn_question_encoder, cnn_paragraph_encoder, tokenizer,
                          hyper_params['max_question_len'], hyper_params['max_paragraph_len'],
                          debug)
    elif hyper_params['model']['name'] == 'rnn':
        rnn_question_encoder = RNNEncoder(hyper_params, tokenizer.vocab_size)
        rnn_paragraph_encoder = RNNEncoder(hyper_params, tokenizer.vocab_size)
        model = Retriever(rnn_question_encoder, rnn_paragraph_encoder, tokenizer,
                          hyper_params['max_question_len'], hyper_params['max_paragraph_len'],
                          debug)
    else:
        raise ValueError('model name {} not supported'.format(hyper_params['model']['name']))
    return model

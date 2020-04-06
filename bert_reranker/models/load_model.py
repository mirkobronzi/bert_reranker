from bert_reranker.models.bert_encoder import BertEncoder
from bert_reranker.models.cnn_model import CNNEncoder
from bert_reranker.models.retriever import Retriever
from bert_reranker.utils.hp_utils import check_and_log_hp


def load_model(hyper_params, tokenizer, debug):
    check_and_log_hp(['name'], hyper_params['model'])
    if hyper_params['model']['name'] == 'bert_encoder':
        bert_question_encoder = BertEncoder(hyper_params)
        bert_paragraph_encoder = BertEncoder(hyper_params)
        model = Retriever(bert_question_encoder, bert_paragraph_encoder, tokenizer,
                          hyper_params['max_question_len'], hyper_params['max_paragraph_len'],
                          debug)
    elif hyper_params['model']['name'] == 'cnn':
        cnn_question_encoder = CNNEncoder(hyper_params, tokenizer.vocab_size)
        cnn_paragraph_encoder = CNNEncoder(hyper_params, tokenizer.vocab_size)
        model = Retriever(cnn_question_encoder, cnn_paragraph_encoder, tokenizer,
                          hyper_params['max_question_len'], hyper_params['max_paragraph_len'],
                          debug)
    else:
        raise ValueError('model name {} not supported'.format(hyper_params['model']['name']))
    return model



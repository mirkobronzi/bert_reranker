from transformers import AutoModel

from bert_reranker.models.bert_encoder import BertEncoder
from bert_reranker.models.retriever import Retriever
from bert_reranker.utils.hp_utils import check_and_log_hp


def load_model(hyper_params, tokenizer, debug):
    check_and_log_hp(['name'], hyper_params['model'])
    if hyper_params['model']['name'] == 'bert_encoder':
        question_encoder, paragraph_encoder = load_bert_encoder_model(hyper_params)
        model = Retriever(question_encoder, paragraph_encoder, tokenizer,
                          hyper_params['max_question_len'], hyper_params['max_paragraph_len'],
                          debug)
    else:
        raise ValueError('model name {} not supported'.format(hyper_params['model']['name']))
    return model


def load_bert_encoder_model(hyper_params):
    model_hparams = hyper_params['model']
    check_and_log_hp(
        ['bert_base', 'layers_pre_pooling', 'layers_post_pooling', 'dropout',
         'normalize_bert_encoder_result', 'dropout_bert', 'freeze_bert', 'pooling_type'],
        model_hparams)
    bert_question = AutoModel.from_pretrained(model_hparams['bert_base'])
    bert_paragraph = AutoModel.from_pretrained(model_hparams['bert_base'])

    bert_question_encoder = BertEncoder(bert_question, hyper_params['max_question_len'],
                                        model_hparams['freeze_bert'], model_hparams['pooling_type'],
                                        model_hparams['layers_pre_pooling'],
                                        model_hparams['layers_post_pooling'],
                                        model_hparams['dropout'],
                                        model_hparams['normalize_bert_encoder_result'],
                                        model_hparams['dropout_bert'])
    bert_paragraph_encoder = BertEncoder(bert_paragraph, hyper_params['max_paragraph_len'],
                                         model_hparams['freeze_bert'],
                                         model_hparams['pooling_type'],
                                         model_hparams['layers_pre_pooling'],
                                         model_hparams['layers_post_pooling'],
                                         model_hparams['dropout'],
                                         model_hparams['normalize_bert_encoder_result'],
                                         model_hparams['dropout_bert'])

    return bert_question_encoder, bert_paragraph_encoder


#!/usr/bin/env python

import argparse
import logging
import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import BertTokenizer, BertModel
from yaml import load

from bert_reranker.data.data_loader import generate_natq_dataloaders
from bert_reranker.models.bert_encoder import BertEncoder
from bert_reranker.models.retriever import Retriever, RetrieverTrainer
from bert_reranker.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)

# TODO
max_question_len_global = 30
max_paragraph_len_global = 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='config file with generic hyper-parameters,  such as optimizer, '
                             'batch_size, ... -  in yaml format', required=True)
    parser.add_argument('--gpu', help='gpu_size', default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config, 'r') as stream:
        hyper_params = load(stream, Loader=yaml.FullLoader)

    check_and_log_hp(
        ['natq_json_file', 'cache_folder', 'batch_size', 'model_name', 'max_question_len',
         'max_paragraph_len', 'embedding_dim'],
        hyper_params)

    os.makedirs(hyper_params['cache_folder'], exist_ok=True)

    model_name = hyper_params['model_name']
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataloader, dev_dataloader = generate_natq_dataloaders(
        hyper_params['natq_json_file'], hyper_params['cache_folder'],
        hyper_params['max_question_len'], hyper_params['max_paragraph_len'],
        tokenizer, hyper_params['batch_size'])

    bert_question = BertModel.from_pretrained(model_name)
    bert_paragraph = BertModel.from_pretrained(model_name)

    bert_question_encoder = BertEncoder(bert_question, max_question_len_global)
    bert_paragraph_encoder = BertEncoder(bert_paragraph, max_paragraph_len_global)

    ret = Retriever(bert_question_encoder, bert_paragraph_encoder, tokenizer,
                    hyper_params['max_question_len'], hyper_params['max_paragraph_len'],
                    hyper_params['embedding_dim'])
    os.makedirs('out', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath='out/{epoch}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )

    # TODO patience
    early_stopping = EarlyStopping('val_acc', mode='max', patience=5)

    trainer = pl.Trainer(
        gpus=args.gpu,
        distributed_backend='dp',
        val_check_interval=0.1,
        min_epochs=1,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping)
    ret_trainee = RetrieverTrainer(ret, train_dataloader, dev_dataloader,
                                   hyper_params['embedding_dim'])
    trainer.fit(ret_trainee)


if __name__ == '__main__':
    main()
